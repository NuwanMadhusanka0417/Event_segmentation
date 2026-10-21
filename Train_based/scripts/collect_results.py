"""One table of every experiment under a results folder (reads each experiment.json written
by summarize_run.py), grouped by experiment id: runs that differ only in SEED are averaged
(mean +- std), so a difference between experiments can be compared with the seed noise.

    python scripts/collect_results.py results            # -> prints + results/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

SEQS = ("13-00", "13-05", "14-03", "14-04", "14-05")


def _ms(v: list[float]) -> str:
    v = [x for x in v if x is not None]
    if not v:
        return "n/a"
    return f"{st.mean(v):.2f}" + (f" ± {st.stdev(v):.2f}" if len(v) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", nargs="?", default="results")
    a = ap.parse_args()
    root = Path(a.root)
    docs = []
    for f in sorted(root.glob("*/experiment.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        doc["_dir"] = f.parent.name
        docs.append(doc)
    # an experiment run at several training sizes (E14) gets one row per size
    sizes: dict[str, set] = defaultdict(set)
    for doc in docs:
        n = doc.get("params", {}).get("SAMPLES")
        if n is not None:                     # runs registered later may not record it
            sizes[doc.get("exp_id", "?")].add(n)
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        exp = doc.get("exp_id", "?")
        n = doc.get("params", {}).get("SAMPLES")
        groups[f"{exp} n={n}" if len(sizes[exp]) > 1 else exp].append(doc)

    lines = ["| exp | runs (seeds) | benchmark 12.5 ms | no temporal | validation | "
             + " | ".join(SEQS) + " | purpose |", "|---" * (len(SEQS) + 6) + "|"]
    for exp in sorted(groups):
        runs = groups[exp]
        done = [r for r in runs if r.get("results") and (r["results"].get("benchmark") or {}).get("12.5ms")]
        seeds = ",".join(str(r["params"].get("SEED", "?")) for r in runs)
        b = [r["results"]["benchmark"]["12.5ms"]["mean"] for r in done]
        bn = [(r["results"].get("benchmark_no_temporal") or {}).get("12.5ms", {}).get("mean") for r in done]
        v = [100 * r["results"]["val_event_iou_best"] for r in done if r["results"].get("val_event_iou_best")]
        per = [_ms([r["results"]["benchmark"]["12.5ms"]["per_sequence"].get(s) for r in done]) for s in SEQS]
        note = runs[0].get("note", "")[:60]
        lines.append(f"| {exp} | {len(done)}/{len(runs)} ({seeds}) | {_ms(b)} | {_ms(bn)} | {_ms(v)} | "
                     + " | ".join(per) + f" | {note} |")
    lines.append("")
    lines.append("Paper (Hua et al. 2025): 77.40 = 82.15 / 76.24 / 79.58 / 73.36 / 75.67. "
                 "Runs still going or failed count as not done (see each folder's EXPERIMENT.md).")
    text = "\n".join(lines)
    print(text)
    (root / "SUMMARY.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
