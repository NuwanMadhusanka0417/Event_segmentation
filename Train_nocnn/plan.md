I'll do a careful audit of our implementation against the paper's equations, then give you a plan. Let me re-read the key files precisely so the analysis is grounded, not from memory.

Read 2 files

I've now audited the encoder, FPE, kernel, cost volume, and estimator against the paper's equations. Your instincts are right on two important points. Here's the full analysis and a prioritized plan. No code changed.

1. Audit — what's faithful vs. not
Component	Paper	Our code	Verdict
VFA kernel eigen-basis (kernel.py)	K = D∗Gaussian, Gaussian similarity	Correct Gram + top-r PCA eigen-filters	✅ faithful (the "eigen part")
FPE application (encoder)	FPE encodes positions D=Xˣ⊙Yʸ (Eq 2,4)	FPE encodes filter-response magnitudes	❌ not faithful
Two-time cost volume (Eq 10)	cos(F0, Fk) over M×M	cross_cost_volume = complex cosine	✅ faithful
Multi-scale sum (Eq 11)	F0↔F1/F2/F4, pool+upsample+sum	same	✅ faithful
Prob-volume threshold (Eq 12)	max(C−α·max−(1−α)·mean,0), normalize	exact match	✅ faithful
Soft-argmax / template (Eq 13–14)	U=ΣP·T, T scaled by 4/Δt	ΣP·offset, vel_scale=1	⚠️ correct form, pixel units not velocity
Velocity ↔ Φ combination	(not in paper — our addition)	raw concat of 3 scalars with Φ	❌ weak, your point

So: cost volume and Eq 12–14 are applied correctly. The two real problems are exactly the ones you flagged — the FPE in the encoder, and how velocity is combined with Φ.

2. Problem A — the FPE in the encoder is misapplied

Look at encoder.py line 85:

code = fpe(phases_x, feat) * fpe(phases_y, feat)      # feat = filter response

Two things are wrong vs. the paper:

It FPE-encodes the response magnitude feat, not a position. The paper FPE-encodes the integer coordinates (Δx,Δy) to build a spatial kernel. Ours encodes "how strong is filter r here" — a different quantity.
The φx·φy double-bind is redundant. Since both use the same scalar feat, exp(i·feat·φx)·exp(i·feat·φy) = exp(i·feat·(φx+φy)) — it collapses to a single FPE. Binding two axes only makes sense when you bind two different values (x and y). Here there's no x/y distinction, so the binding does nothing meaningful.

Consequence: cosine(F(x), F(x′)) does not equal the paper's Gaussian patch-kernel similarity. It's a magnitude-closeness kernel with spurious cross-terms → the cost volume matches are less discriminative than the paper's.

The correct construction (and it keeps the eigen speed): by Mercer's theorem the kernel patch-similarity equals Σ_r feat_r(x)·feat_r(x′). To realize that as a hypervector whose cosine reproduces it, bind each eigen-response to a fixed random role vector e_r and bundle:

F(x) = Σ_r  feat_r(x) · e_r          (e_r = random quasi-orthogonal hypervectors)

Then ⟨F(x),F(x′)⟩ = Σ_r feat_r(x)feat_r(x′) = the true kernel similarity. This is the faithful, correct hypervector encoder — a small change (replace FPE-of-magnitude with weight×role bundling).

3. Problem B — velocity is combined with Φ incorrectly (your main point)

Current hdems.py → paper_features:

feats = cat([Φ.real, Φ.imag, residual_vx, residual_vy, |r|])   # 3 raw numbers tacked on

Problems:

Not VSA-structured — velocity is 3 raw scalars glued to 1024 Φ numbers, so it's heavily under-represented and has no topological similarity (similar velocities don't get similar codes).
The x/y motion isn't given distinct roles.

Your proposed fix is right in spirit. The correct VSA construction:

Vx = FPE_gaussian(vx ; φ_vx)          # velocity-x hypervector (topological)
Vy = FPE_gaussian(vy ; φ_vy)          # velocity-y hypervector
Mv = Vx ⊙ Vy                          # bind x,y roles -> one 2-D velocity code

Now similar velocities → similar Mv (Gaussian FPE), and x vs y motion are cleanly separated by their own bases.

One important nuance for how to give it to the classifier. You said "bind x,y with Φ." There are two options, and the choice depends on the readout:

Concatenate blocks [Φ | Mv] (appearance block + motion block) — recommended for the Ridge (linear) readout, because a linear classifier can read each block directly. This turns 3 weak scalars into a rich, high-dimensional, topologically-smooth motion feature → much better linear separability.
Fully bind Φ ⊙ Mv into one hypervector — this is "purer VSA," but binding entangles motion and appearance, and a linear readout can't easily disentangle them. It's the right move only if you later switch to a VSA-native readout (resonator/cleanup memory), not a linear one.

So: encode velocity as a hypervector and bind vx↔vy (yes) — but provide it alongside Φ as a concatenated block for the linear readout, rather than binding everything into a single vector. (Optionally add Φ⊙Mv as an extra interaction block if you want.)

4. Prioritized plan to raise accuracy

P1 — Fix the encoder (Problem A). Replace fpe(φx,feat)*fpe(φy,feat) with the random-role bundling F(x)=Σ_r feat_r·e_r (or the faithful position-code encoder Σ_Δ T(x+Δ)·Xᐩˣ⊙Yᐩʸ). This makes cosine(F,F′) = the true kernel similarity → sharper cost-volume peaks → better flow. Highest impact (affects every match).

P2 — Proper velocity encoding (Problem B). Build Mv = FPE(vx)⊙FPE(vy) with Gaussian bases + a tunable velocity bandwidth (scale residual velocity first, like the GVFA project's VEL_BW). Feed the Ridge readout [Φ | Mv] instead of [Φ | 3 scalars]. High impact for the readout.

P3 — Estimator faithfulness (small). (a) Turn on the cost-volume pre-smoothing (smooth>1, the avg-pool in Eq 12) to denoise peaks. (b) Optionally scale the template by 4/Δt for true velocity units (only matters if you want physical velocity; ego-residual is scale-agnostic so low priority).

P4 — Polarity role-binding (minor faithfulness). The encoder sums polarities before FPE; the paper role-binds them (F₊⊙R₊+F₋⊙R₋, Eq 8). Low priority (you'd dropped polarity earlier), but it's a known gap.

P5 — Ego-residual (tuning, not correctness). Affine + IRLS is reasonable; if background still leaks, consider a richer motion model or per-region fit. Lowest priority.

5. How to know it worked (before you build)

These are hypotheses — validate empirically:

Fit Ridge on a small --max-train-samples set before and after each change; compare eval mIoU and fill docs/RIDGE_RESULTS.md.
Sanity-check P1 by visualizing the decoded flow (out["flow"]) — sharper, cleaner flow after the encoder fix.
Do P1 and P2 one at a time so you can attribute the gain.

Order I recommend: P1 → measure → P2 → measure → P3. P1 and P2 are the two you correctly identified; they're the ones most likely to move accuracy.

Want me to turn P1 and P2 into a concrete implementation spec (exact functions to change, the new tensor shapes, and the config knobs) — still without editing — so you can review before I build it?