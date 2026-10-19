"""Describe one experiment inside its results folder: EXPERIMENT.md (for people) and
experiment.json (for collect_results.py).

  init    at the start of the job: experiment id, purpose, what to compare it with, every
          setting of the run
  finish  at the end: the results -- validation score, chosen threshold / temporal beta,
          benchmark per sequence (12.5 / 50 ms, with and without the temporal filter),
          motion / appearance ablation, object metrics, latency, head size

    python scripts/summarize_run.py init   RESULTS_DIR --exp-id E0 --note "..." \\
        --compare "..." --param HEAD=mfunet --param SEED=0 ...
    python scripts/summarize_run.py finish RESULTS_DIR
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

SEQS = ("13-00", "13-05", "14-03", "14-04", "14-05")
PAPER = {"13-00": 82.15, "13-05": 76.24, "14-03": 79.58, "14-04": 73.36, "14-05": 75.67}


def _load(d: Path) -> dict:
    f = d / "experiment.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _save(d: Path, doc: dict) -> None:
    (d / "experiment.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def init(d: Path, exp_id: str, note: str, compare: str, params: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    p = dict(kv.split("=", 1) for kv in params)
    doc = {"exp_id": exp_id, "note": note, "compare_with": compare, "params": p,
           "started": datetime.now().isoformat(timespec="seconds"), "results": None}
    _save(d, doc)
    lines = [f"# Experiment {exp_id}", "", f"**Purpose:** {note}", "",
             f"**Compare with:** {compare}", "",
             f"Started {doc['started']}  |  job {p.get('JOB', '?')}", "",
             "## Settings (run_segmentation.pbs parameters)", "", "| parameter | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in p.items()]
    lines += ["", "Full configuration: run_config.txt and evimo_seg.yaml in this folder.", "",
              "## Results", "", "_(written when the job finishes)_", ""]
    (d / "EXPERIMENT.md").write_text("\n".join(lines), encoding="utf-8")


def _first(pat: str, text: str, cast=str):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def _bench(d: Path, name: str) -> dict | None:
    f = d / name
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for w in j["windows"]:
        out[f"{w['window_ms']:g}ms"] = {"mean": w["mean"], "per_sequence": w["per_sequence"]}
    out["setting"] = {k: j["setting"].get(k) for k in ("decision_threshold", "temporal_beta", "oracles")}
    return out


def finish(d: Path) -> None:
    doc = _load(d)
    logs = sorted(p for p in d.iterdir() if re.search(r"\.o(\d+|local\d+)$", p.name))   # PBS job log
    log = logs[0].read_text(encoding="utf-8", errors="replace") if logs else ""
    model_eval, _, control = log.partition("EVENTS-ONLY BASELINE")      # the control runs last
    vals = [float(v) for v in re.findall(r"val_eventIoU=([0-9.]+)", log)]
    r = {
        "val_event_iou_best": max(vals) if vals else None,
        "val_curve": vals,
        "chosen": _first(r"\[train\] chosen: (.*)", log),
        "best_epoch": _first(r"best epoch=(\d+)", log, int),
        "train_frames": _first(r"sample cap: train (\d+)/", log, int),
        "fg_iou_ours_eval": _first(r"FG IoU \(moving vs rest\) \[HEADLINE\]: ([0-9.]+)", model_eval, float),
        "ablation": _first(r"FG IoU   full (.*)", model_eval),
        "verdict": _first(r"Verdict: (.*)", model_eval),
        "object_miou": _first(r"Object mIoU \(matched\):\s+([0-9.]+)", model_eval, float),
        "object_miou_gt_pixels": _first(r"best case\]: ([0-9.]+)", model_eval, float),
        "static_false_object": _first(r"Static frames with a false object: ([0-9.]+%)", model_eval),
        "latency_ms": _first(r"Latency:\s+([0-9.]+) ms", model_eval, float),
        "head_params": _first(r"Head params: ([0-9,]+)", model_eval),
        "control_fg_iou": _first(r"FG IoU \(moving vs rest\) \[HEADLINE\]: ([0-9.]+)", control, float),
        "elapsed": _first(r"elapsed (\d+h \d+m)", log),
        "benchmark": _bench(d, "benchmark_hua2025.json"),
        "benchmark_no_temporal": _bench(d, "benchmark_hua2025_no_temporal.json"),
        "benchmark_oracle_vicon": _bench(d, "benchmark_hua2025_oracle_vicon.json"),
        "errors": re.findall(r"(Traceback.*|ERROR.*)", log)[:3],
    }
    doc["results"] = r
    doc["finished"] = datetime.now().isoformat(timespec="seconds")
    _save(d, doc)

    md = (d / "EXPERIMENT.md").read_text(encoding="utf-8") if (d / "EXPERIMENT.md").exists() else ""
    md = md.split("## Results")[0]
    out = ["## Results", "", f"Finished {doc['finished']}  |  elapsed {r['elapsed']}", ""]
    if r["errors"]:
        out += ["**ERRORS in the log:**", ""] + [f"    {e}" for e in r["errors"]] + [""]
    b = r["benchmark"]
    if b:
        out += ["### Benchmark (Hua et al. 2025 protocol: event IoU %, frames with a mover)", "",
                "| window | mean | " + " | ".join(SEQS) + " |", "|---" * (len(SEQS) + 2) + "|"]
        for tag, bb in (("", b), (" no temporal", r["benchmark_no_temporal"]),
                        (" Vicon rotation (oracle)", r["benchmark_oracle_vicon"])):
            if not bb:
                continue
            for w in ("12.5ms", "50ms"):
                if w in bb:
                    ps = bb[w]["per_sequence"]
                    out.append(f"| {w}{tag} | **{bb[w]['mean']:.2f}** | "
                               + " | ".join(f"{ps.get(s, float('nan')):.1f}" for s in SEQS) + " |")
        out.append("| paper | 77.40 | " + " | ".join(f"{PAPER[s]:.1f}" for s in SEQS) + " |")
        out += ["", f"Decision threshold {b['setting']['decision_threshold']}, temporal beta "
                    f"{b['setting']['temporal_beta']}, inputs {b['setting']['oracles']}", ""]
    out += ["### Training / validation", "",
            f"- train frames {r['train_frames']}, best epoch {r['best_epoch']}",
            f"- best validation event IoU {r['val_event_iou_best']}  ({r['chosen']})", "",
            "### Our eval (test split, pixels)", "",
            f"- FG IoU {r['fg_iou_ours_eval']} (events-only control {r['control_fg_iou']})",
            f"- ablation: {r['ablation']}",
            f"- verdict: {r['verdict']}",
            f"- object mIoU {r['object_miou']} (grouping on ground-truth moving pixels: "
            f"{r['object_miou_gt_pixels']}); static frames with a false object {r['static_false_object']}",
            f"- latency {r['latency_ms']} ms/frame, head parameters {r['head_params']}", ""]
    (d / "EXPERIMENT.md").write_text(md + "\n".join(out), encoding="utf-8")
    print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["init", "finish"])
    ap.add_argument("results_dir")
    ap.add_argument("--exp-id", default="E?")
    ap.add_argument("--note", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--param", action="append", default=[])
    a = ap.parse_args()
    d = Path(a.results_dir)
    if a.step == "init":
        init(d, a.exp_id, a.note, a.compare, a.param)
    else:
        finish(d)


if __name__ == "__main__":
    main()
