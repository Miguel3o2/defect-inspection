"""Collect REAL results from artifacts/ and reports/ into reports/RESULTS.md, and render the cold email from them.

  python scripts/collect_results.py                 # writes reports/RESULTS.md
  python scripts/collect_results.py --email         # also renders reports/email_draft.md

Nothing is invented: a missing artifact shows as "not run", and the email refuses to render while any numeric
placeholder cannot be filled from real results. Run it in the directory where you ran the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _f1(report: dict | None) -> float | None:
    return float(report["macro avg"]["f1-score"]) if report else None


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _missing(what: str, cmd: str) -> str:
    return f"_Not run yet: {what}. Run `{cmd}`._"


def headline(art: Path, facts: dict) -> str:
    rows = [("HOG+LBP+GLCM + LogReg", "logreg"), ("HOG+LBP+GLCM + LightGBM", "lightgbm")]
    table = [("Model", "Val macro-F1", "Test macro-F1")]
    for label, key in rows:
        v, t = _f1(_json(art / f"report_{key}_val.json")), _f1(_json(art / f"report_{key}_test.json"))
        table.append((label, _fmt(v), _fmt(t)))
        vc, tc = _f1(_json(art / f"report_{key}_val_cleaned.json")), _f1(_json(art / f"report_{key}_test_cleaned.json"))
        if vc is not None or tc is not None:
            table.append((f"{label} (label-noise cleaned train)", _fmt(vc), _fmt(tc)))
        facts[f"baseline_{key}_val_f1"], facts[f"baseline_{key}_test_f1"] = v, t
    dv, dt = _f1(_json(art / "report_deep_val.json")), _f1(_json(art / "report_deep_test.json"))
    table.append(("ResNet fine-tuned", _fmt(dv), _fmt(dt)))
    facts.update(deep_val_f1=dv, deep_test_f1=dt, baseline_test_f1=facts["baseline_lightgbm_test_f1"])
    if all(r[1] == "n/a" and r[2] == "n/a" for r in table[1:]):
        return _missing("baseline and deep models", "make baseline && make deep")
    out = ["| " + " | ".join(r) + " |" for r in table]
    out.insert(1, "|---|---|---|")
    if dt is None and facts["baseline_lightgbm_test_f1"] is None:
        out.append("\n_Test column empty: run each trainer once with `--eval-test` at the very end. "
                   "Quote validation numbers only as validation numbers._")
    return "\n".join(out)


def label_noise(art: Path, rep: Path, facts: dict) -> str:
    issues, confirmed = rep / "label_issues.csv", rep / "label_issues_confirmed.csv"
    if not issues.exists():
        return _missing("label-noise detection", "make label-noise")
    n = len(list(csv.DictReader(issues.open(encoding="utf-8"))))
    n_conf = len(list(csv.DictReader(confirmed.open(encoding="utf-8")))) if confirmed.exists() else None
    v, vc = _f1(_json(art / "report_lightgbm_val.json")), _f1(_json(art / "report_lightgbm_val_cleaned.json"))
    lines = [f"- cleanlab flagged **{n}** candidate samples" + (f"; **{n_conf}** kept after manual review." if n_conf is not None
                                                                 else "; manual review not done yet (`reports/label_issues_confirmed.csv`).")]
    if v is not None and vc is not None:
        lines.append(f"- LightGBM validation macro-F1: {v:.3f} -> {vc:.3f} after retraining on the cleaned train split "
                     f"(Δ {vc - v:+.3f}; validation labels untouched, so the gain is understated if val is also noisy).")
        facts["noise_delta"] = vc - v
    return "\n".join(lines)


def deployment(rep: Path, facts: dict) -> str:
    bench = _json(rep / "benchmark.json")
    if not bench:
        return _missing("benchmark", "make benchmark")
    rows = {r["backend"]: r for r in bench["rows"]}
    env = bench["env"]
    md = _text(rep / "benchmark.md") or ""
    torch_, ort_, int8 = rows.get("pytorch fp32"), rows.get("onnxruntime fp32"), rows.get("onnxruntime int8")
    if torch_ and ort_:
        facts.update(torch_p50_ms=torch_["p50_ms"], ort_p50_ms=ort_["p50_ms"],
                     latency_reduction_pct=100 * (torch_["p50_ms"] - ort_["p50_ms"]) / torch_["p50_ms"])
    notes = []
    if torch_ and ort_:
        notes.append(f"- ONNX Runtime p50 latency {ort_['p50_ms']:.2f} ms vs PyTorch {torch_['p50_ms']:.2f} ms "
                     f"({facts['latency_reduction_pct']:+.0f}% change; {env['threads']} thread(s), {env['cpu_count']} cores available).")
    if ort_ and int8:
        d = (int8.get("macro_f1") - ort_.get("macro_f1")) if "macro_f1" in int8 and "macro_f1" in ort_ else None
        notes.append(f"- int8: {ort_['size_mb']:.1f} -> {int8['size_mb']:.1f} MB, p50 {ort_['p50_ms']:.2f} -> {int8['p50_ms']:.2f} ms"
                     + (f", val macro-F1 {d:+.3f} vs fp32." if d is not None else "; accuracy not measured (--skip-accuracy)."))
        facts["int8_f1_delta"] = d
    if not torch_:
        notes.append("- No PyTorch row: rerun the benchmark with `--ckpt artifacts/resnet_best.pt`.")
    return md + "\n" + "\n".join(notes)


def patchcore(art: Path, facts: dict) -> str:
    table = _text(art / "patchcore" / "results.md")
    if not table:
        return _missing("PatchCore", "make patchcore")
    cats = [_json(p) for p in sorted((art / "patchcore").glob("*/results.json"))]
    if cats:
        facts["patchcore_image_auroc"] = sum(c["image_auroc"] for c in cats) / len(cats)
        facts["patchcore_pixel_auroc"] = sum(c["pixel_auroc"] for c in cats) / len(cats)
        facts["patchcore_n_categories"] = len(cats)
    return table


def provenance() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown (not a git checkout)"
    return f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} from artifacts/ and reports/ at commit {sha}._"


def claims(facts: dict) -> list[str]:
    out = []
    b, d = facts.get("baseline_lightgbm_test_f1"), facts.get("deep_test_f1")
    if b is not None and d is not None:
        out.append(f"On the held-out test split, LightGBM on handcrafted features scores {b:.3f} macro-F1 and the "
                   f"fine-tuned ResNet {d:.3f} (difference {d - b:+.3f}).")
    if "latency_reduction_pct" in facts:
        out.append(f"On CPU, ONNX Runtime p50 latency is {facts['ort_p50_ms']:.1f} ms vs {facts['torch_p50_ms']:.1f} ms in PyTorch.")
    if "patchcore_image_auroc" in facts:
        out.append(f"PatchCore reaches a mean image AUROC of {facts['patchcore_image_auroc']:.3f} over "
                   f"{facts['patchcore_n_categories']} MVTec AD categories.")
    return out or ["Nothing yet: no complete results found."]


def collect(root: Path) -> tuple[str, dict]:
    art, rep = root / "artifacts", root / "reports"
    facts: dict = {}
    parts = [
        ("Headline results (NEU-DET classification)", headline(art, facts)),
        ("Augmentation ablation", _text(art / "ablation.md") or _missing("ablation", "make ablate")),
        ("Label noise", label_noise(art, rep, facts)),
        ("Anomaly detection (PatchCore, MVTec AD)", patchcore(art, facts)),
        ("PatchCore vs supervised on unseen defect types",
         _text(art / "comparison" / "results.md") or _missing("comparison", "make compare")),
        ("Deployment benchmark", deployment(rep, facts)),
        ("Drift monitoring simulation",
         _text(rep / "drift" / "drift_simulation.md") or _missing("drift simulation", "make drift-simulate")),
    ]
    md = ["# Results", "", provenance(), ""]
    for title, body in parts:
        md += [f"## {title}", "", body, ""]
    md += ["## Claims supported by these numbers", "", *[f"- {c}" for c in claims(facts)], ""]
    return "\n".join(md), facts


_FMT = {"latency_reduction_pct": ".0f", "ort_p50_ms": ".1f", "torch_p50_ms": ".1f"}


def render_email(template: str, facts: dict) -> str:
    keys = sorted(set(re.findall(r"\{\{(\w+)\}\}", template)))
    missing = [k for k in keys if facts.get(k) is None]
    if missing:
        raise SystemExit("Cannot render the email: no real value yet for " + ", ".join(missing) +
                         ". Run the missing stages (see RUNBOOK.md); do not type numbers in by hand.")
    out = template
    for k in keys:
        out = out.replace("{{" + k + "}}", format(facts[k], _FMT.get(k, ".3f")))
    left = sorted(set(re.findall(r"\[[^\]]+\]", out)))
    print("Fill in by hand before sending: " + (", ".join(left) or "nothing"), file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/RESULTS.md")
    ap.add_argument("--email", action="store_true")
    ap.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "docs" / "cold_email_template.md"))
    a = ap.parse_args(argv)
    md, facts = collect(Path(a.root))
    out = Path(a.root) / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    if a.email:
        draft = Path(a.root) / "reports" / "email_draft.md"
        draft.write_text(render_email(Path(a.template).read_text(encoding="utf-8"), facts), encoding="utf-8")
        print(f"wrote {draft}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
