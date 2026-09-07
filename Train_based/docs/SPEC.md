# HD-EMS — Build Specification
### Hyperdimensional Event Motion Segmentation · CVPR target · Cursor-ready

> Put this file at `docs/SPEC.md` in your repo. It is written to be read by both you and Cursor.

---

## 0. Corrections to the previous plan (read first)

I numerically verified the core mechanism. Two results change the design:

**✅ CONFIRMED — the separable construction is exact.**
Building the bundled field `Φ(x) = Σ_v F(x+v) ∘ P(v)` via two 1-D passes instead of M² shifts matches the explicit sum to **5.3e-16 relative error**. This is not an approximation — FPE position codes factor exactly as `exp(i·vx·φˣ)·exp(i·vy·φʸ)` in the phasor domain, so the 2-D binding-convolution is *exactly* rank-1 separable. The efficiency mechanism is mathematically sound.

**❌ CORRECTED — bundling capacity collapses at large M.** My earlier claim that smoothness rescues large windows was wrong. Measured retrieval (d=1024, argmax endpoint error in px):

| M | items | σ=0 | σ=1 | σ=2 | σ=3 |
|---|---|---|---|---|---|
| 7 | 49 | c=0.84, e=0.0 | c=0.95, **e=0.0** | c=0.93, e=1.0 | c=0.71, e=1.0 |
| 11 | 121 | c=0.52, e=0.0 | c=0.74, e=1.4 | c=0.84, e=1.0 | c=0.93, e=3.2 |
| 15 | 225 | c=0.62, e=6.3 | c=0.38, e=5.1 | c=0.71, e=1.0 | c=0.75, **e=4.1** |

Smoothing helps a little; it does not save you. **M=31 in a single bundle will not work.**

**The fix — hierarchical bundling.** Use M=7 per level (inside the validated regime) across a 4-level pyramid. Range compounds geometrically while every level stays high-fidelity:

| level | resolution | range | cost |
|---|---|---|---:|
| 0 | 120×160 | ±3 px | 0.2753 GMAC |
| 1 | 60×80 | ±6 px | 0.0688 |
| 2 | 30×40 | ±12 px | 0.0172 |
| 3 | 15×20 | ±24 px | 0.0043 |
| **total** | | **±45 px** | **0.366 GMAC** |

vs. explicit cost volume at the same range: **111× cheaper**. vs. RAFT all-pairs at 1/8: **16× cheaper**. The pyramid costs only 33% more than level 0 alone, so the efficiency claim survives fully — and now it is *validated* rather than assumed.

This is also a friendlier design for reviewers: coarse-to-fine is exactly what PWC-Net and RAFT do.

---

## 1. Using VSA for accuracy, not only speed

Your previous plan used VSA purely as a compression trick. That is a weak CVPR story. Here are four mechanisms where VSA should **improve accuracy** — these become your accuracy claims.

### A1. Large displacement at constant cost
Cost per level is independent of the *number* of candidate displacements — it depends only on M taps and d channels. Combined with the pyramid, you get ±45 px search for 0.37 GMAC. Large-displacement flow is a known hard problem, and the source paper itself reports VSA-Flow is *strongest* at large flow (dt=4 on MVSEC: EPE 1.44 vs MultiCM 1.69). **Claim: better accuracy on fast motion, where event cameras matter most.**

### A2. Distributional matching instead of early argmax
`Φ(x)` encodes the *entire* matching function over displacement, not its maximum. Conventional pipelines argmax or softmax the cost volume early and destroy multimodality. Keeping the full function as a hypervector lets the decoder resolve ambiguity itself. **Claim: better behaviour under the aperture problem, repetitive texture, and at occlusion boundaries** — exactly where flow-based segmentation normally fails.

### A3. Temporal binding — the big one for segmentation
This directly fixes the weakness the source paper admits to (*"comparatively weaker in exploiting temporal information"*). Bind time as a role:

```
Traj(x) = Σ_t  F^t(x) ∘ T^t          T^t = FPE code of time t
```

One hypervector now holds a location's whole spatio-temporal trajectory; unbinding with `T^t` queries any instant. Segment on **trajectory hypervectors** rather than instantaneous flow. Trajectory-level grouping is substantially more robust than per-frame flow clustering. **Claim: better temporal consistency and fewer identity switches.**

### A4. Exact equivariance as inductive bias
FPE is *exactly* translation-equivariant by construction; a CNN only approximates it. Flow is fundamentally about translation, so this is the correct prior. **Claim: better generalization, especially cross-dataset and in low-data regimes** — test by training on DSEC and evaluating on MVSEC without fine-tuning. This is the ablation reviewers will find most convincing about "why VSA and not just a ConvNet."

> Together with the efficiency mechanisms, the paper's claim becomes **"more accurate *and* cheaper,"** not "cheaper but comparable." That is the difference between a CVPR accept and a workshop paper.

---

## 2. Final architecture

```
Events ─▶ per-polarity accumulative Time Surfaces, 4-level pyramid       [0 params]
            │
            ▼
        Rank-r analytic VFA encoder  (separable, eigen-basis, r≈64)      [0 params]  §A4
            │   F_l ∈ R^{r×H_l×W_l}
            ▼
        Hierarchical bundled matching field  Φ_l = Σ_v F_l(x+v)∘P(v)     [0 params]  §A1,A2
            │   M=7 per level, separable 1-D passes, d≈1024
            ▼
        Temporal binding  Traj = Σ_t F^t ∘ T^t                           [0 params]  §A3
            │
            ▼
        Coarse-to-fine decoder: slim GRU / 3–4 conv per level            [TRAINED ~1M]
            │
            ├──▶ optical flow U
            └──▶ motion embedding ──▶ segmentation head                  [TRAINED ~0.5M]
                                          │
                                          ▼
                                   VSA prototype bundling (temporal ID)  [0 params]  §A3
```

**Parameter budget:** ~1.5 M trained vs E-RAFT's ~5.3 M. Encoder, cost volume and temporal memory are all zero-parameter.

---

## 3. Repository structure

```
hdems/
├── docs/SPEC.md                 ← this file
├── .cursorrules                 ← §6
├── configs/
│   ├── base.yaml
│   ├── dsec_flow.yaml
│   └── evimo_seg.yaml
├── hdems/
│   ├── vsa/
│   │   ├── fpe.py               # FPE, binding, bundling      [CRITICAL]
│   │   ├── kernel.py            # VFA kernel + eigen-basis    [CRITICAL]
│   │   ├── field.py             # separable bundled field     [CRITICAL]
│   │   └── temporal.py          # time binding, prototypes
│   ├── data/
│   │   ├── time_surface.py
│   │   ├── dsec.py
│   │   ├── evimo.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── encoder.py           # analytic VSA encoder
│   │   ├── matching.py          # hierarchical field
│   │   ├── decoder.py           # trained head
│   │   ├── segmentation.py
│   │   └── hdems.py             # assembly
│   ├── losses/{flow.py,seg.py}
│   ├── train.py
│   ├── eval.py
│   └── benchmark.py             # FLOPs / params / latency
├── tests/
│   ├── test_fpe.py              # MUST PASS FIRST
│   ├── test_kernel_rank.py
│   ├── test_field_exact.py      # separability == explicit
│   ├── test_retrieval.py        # the go/no-go gate
│   └── test_equivariance.py
└── scripts/{prepare_dsec.sh,prepare_evimo.sh}
```

---

## 4–9. See full specification in repository README and .cursorrules.
