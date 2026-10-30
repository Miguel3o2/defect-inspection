import json

import pytest

from scripts import collect_results as cr


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _report(f1):
    return {"macro avg": {"f1-score": f1}, "accuracy": f1}


@pytest.fixture
def full(tmp_path):
    a, r = tmp_path / "artifacts", tmp_path / "reports"
    _w(a / "report_lightgbm_val.json", _report(0.90))
    _w(a / "report_lightgbm_test.json", _report(0.88))
    _w(a / "report_lightgbm_val_cleaned.json", _report(0.92))
    _w(a / "report_logreg_val.json", _report(0.85))
    _w(a / "report_deep_val.json", _report(0.97))
    _w(a / "report_deep_test.json", _report(0.95))
    _w(r / "label_issues.csv", "path,given_label\nx,a\ny,b\nz,c\n")
    _w(r / "label_issues_confirmed.csv", "path,given_label\nx,a\n")
    _w(a / "patchcore" / "results.md", "| Category | Image AUROC |\n|---|---|\n| bottle | 0.99 |")
    _w(a / "patchcore" / "bottle" / "results.json", {"image_auroc": 0.99, "pixel_auroc": 0.95})
    _w(a / "patchcore" / "capsule" / "results.json", {"image_auroc": 0.93, "pixel_auroc": 0.97})
    _w(r / "benchmark.md", "| Backend | p50 |\n|---|---|")
    _w(r / "benchmark.json", {"env": {"threads": 1, "cpu_count": 4}, "rows": [
        {"backend": "pytorch fp32", "size_mb": 45.0, "p50_ms": 20.0, "macro_f1": 0.95},
        {"backend": "onnxruntime fp32", "size_mb": 44.7, "p50_ms": 12.0, "macro_f1": 0.95},
        {"backend": "onnxruntime int8", "size_mb": 11.5, "p50_ms": 7.0, "macro_f1": 0.94}]})
    return tmp_path


def test_empty_project_reports_not_run_and_no_claims(tmp_path):
    md, facts = cr.collect(tmp_path)
    assert md.count("Not run yet") >= 5
    assert "Nothing yet" in md and "latency_reduction_pct" not in facts


def test_numbers_and_derived_values_are_correct(full):
    md, facts = cr.collect(full)
    assert facts["latency_reduction_pct"] == pytest.approx(40.0)          # (20 - 12) / 20
    assert facts["patchcore_image_auroc"] == pytest.approx(0.96) and facts["patchcore_n_categories"] == 2
    assert facts["noise_delta"] == pytest.approx(0.02) and facts["int8_f1_delta"] == pytest.approx(-0.01)
    assert "| ResNet fine-tuned | 0.970 | 0.950 |" in md
    assert "flagged **3** candidate samples" in md and "**1** kept after manual review" in md
    assert "difference +0.070" in md
    assert "12.0 ms vs 20.0 ms in PyTorch" in md                            # ONNX Runtime first, then PyTorch


def test_email_uses_only_real_numbers(full, tmp_path):
    _, facts = cr.collect(full)
    out = cr.render_email((cr.Path(cr.__file__).resolve().parents[1] / "docs" / "cold_email_template.md").read_text(), facts)
    assert "{{" not in out
    assert "0.880 test macro-F1" in out and "0.950 for a fine-tuned ResNet" in out and "(40% lower)" in out
    assert "[Name]" in out                                                  # personal fields stay manual


def test_email_refuses_when_numbers_missing(tmp_path):
    _, facts = cr.collect(tmp_path)
    with pytest.raises(SystemExit) as e:
        cr.render_email("Score {{deep_test_f1}} and {{ort_p50_ms}}", facts)
    assert "deep_test_f1" in str(e.value) and "ort_p50_ms" in str(e.value)


def test_cli_writes_files(full):
    assert cr.main(["--root", str(full), "--email"]) == 0
    assert (full / "reports" / "RESULTS.md").exists() and (full / "reports" / "email_draft.md").exists()
