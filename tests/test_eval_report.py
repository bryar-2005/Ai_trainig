"""Regression tests for reproducible human review reports."""

import pytest

pytest.importorskip("sacrebleu")
pytest.importorskip("sentencepiece")
pytest.importorskip("regex")

from ml.eval.compare import render_comparison
from ml.eval.metrics import compute_metrics
from tests.test_eval_metrics import record


def test_html_escapes_inputs_isolates_direction_and_samples_100_rows():
    records = [
        record(f"pair-{i}", f"<img src=x> source {i}", "سڵاو و بەخێربێیت") for i in range(125)
    ]
    baseline = ["old & unsafe <text>"] * len(records)
    candidate = ["new & unsafe <text>"] * len(records)
    metrics = {
        "baseline": compute_metrics(records, baseline, direction="en-ckb"),
        "candidate": compute_metrics(records, candidate, direction="en-ckb"),
        "regression": False,
        "label": '<img src="bad">',
        "warnings": ["<script>bad()</script>"],
    }
    html, count = render_comparison(records, baseline, candidate, metrics, direction="en-ckb")
    again, _ = render_comparison(records, baseline, candidate, metrics, direction="en-ckb")
    assert count == 100 and html == again
    assert "<img src=x>" not in html and "&lt;img src=x&gt;" in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;img src=&quot;bad&quot;&gt;" in html
    assert "Content-Security-Policy" in html and 'dir="rtl"' in html and 'dir="ltr"' in html
    assert "<del>" in html and "<ins>" in html
    assert html.count('class="cps"') == 200


def test_rtl_controls_are_removed_and_reverse_direction_is_correct():
    records = [record("a", "سڵاو\u202eای", "Hello\u2066 friend\u2069")]
    base, candidate = ["old\u202e"], ["new\u2069"]
    metrics = {
        "baseline": compute_metrics(records, base, direction="ckb-en"),
        "candidate": compute_metrics(records, candidate, direction="ckb-en"),
        "regression": True,
    }
    html, _ = render_comparison(records, base, candidate, metrics, direction="ckb-en")
    assert "\u202e" not in html and "\u2066" not in html and "\u2069" not in html
    assert '<td dir="rtl">سڵاو' in html
    assert '<td dir="ltr">Hello' in html
    assert "REGRESSION" in html
