"""Safe, self-contained side-by-side HTML for human A/B review."""

from __future__ import annotations

import difflib
import html
import random
from typing import Any

from ml.eval.metrics import cue_cps
from ml.eval.records import EvaluationRecord

_BIDI_FORMAT = frozenset("\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _safe_text(value: str) -> str:
    # Do not let input supply HTML or visual bidi overrides; the report sets
    # the direction explicitly on each isolated text cell instead.
    return "".join(
        char for char in value if char not in _BIDI_FORMAT and (char >= " " or char in "\n\t")
    )


def _diff(left: str, right: str) -> tuple[str, str]:
    a, b = _safe_text(left), _safe_text(right)
    before: list[str] = []
    after: list[str] = []
    for operation, a0, a1, b0, b1 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        first, second = html.escape(a[a0:a1]), html.escape(b[b0:b1])
        if operation == "equal":
            before.append(first)
            after.append(second)
        else:
            if first:
                before.append(f"<del>{first}</del>")
            if second:
                after.append(f"<ins>{second}</ins>")
    return "".join(before), "".join(after)


def _score(value: Any) -> str:
    return "not computed" if value is None else f"{float(value):.3f}"


def render_comparison(
    records: list[EvaluationRecord],
    baseline: list[str],
    candidate: list[str],
    metrics: dict[str, Any],
    *,
    direction: str,
    limit: int = 100,
    seed: int = 42,
) -> tuple[str, int]:
    """Return a deterministic, escaped HTML document and displayed row count."""
    if not records or len(records) != len(baseline) or len(records) != len(candidate):
        raise ValueError("Report inputs must have the same nonzero number of rows")
    if direction not in {"en-ckb", "ckb-en"} or limit < 1:
        raise ValueError("Invalid report direction or row limit")
    selected = (
        list(range(len(records)))
        if len(records) <= limit
        else sorted(random.Random(seed).sample(range(len(records)), limit))
    )
    source_dir, target_dir = ("ltr", "rtl") if direction == "en-ckb" else ("rtl", "ltr")
    rows: list[str] = []
    for index in selected:
        record = records[index]
        before, after = _diff(baseline[index], candidate[index])
        context = "\n".join(record.example.context)
        context_markup = (
            "<details><summary>Available previous source context (up to five)</summary>"
            f'<div dir="{source_dir}">{html.escape(_safe_text(context))}</div></details>'
            if context
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{index + 1}<br><small>{html.escape(_safe_text(record.film_id))} / "
            f"{html.escape(_safe_text(record.pair_id))}</small></td>"
            f'<td dir="{source_dir}">{html.escape(_safe_text(record.source))}{context_markup}</td>'
            f'<td dir="{target_dir}">{html.escape(_safe_text(record.reference))}</td>'
            f'<td dir="{target_dir}">{before}<small class="cps">'
            f"{cue_cps(baseline[index], record.duration_ms):.1f} CPS</small></td>"
            f'<td dir="{target_dir}">{after}<small class="cps">'
            f"{cue_cps(candidate[index], record.duration_ms):.1f} CPS</small></td>"
            "</tr>"
        )
    first, second = metrics["baseline"], metrics["candidate"]
    metrics_rows = []
    for name, left, right in (
        ("chrF++", first["chrfpp"], second["chrfpp"]),
        ("spBLEU", (first["spbleu"] or {}).get("score"), (second["spbleu"] or {}).get("score")),
        ("COMET", (first["comet"] or {}).get("score"), (second["comet"] or {}).get("score")),
        (
            "Above 17 CPS (%)",
            first["cps"]["above_target_percent"],
            second["cps"]["above_target_percent"],
        ),
        (
            "Above 21 CPS (%)",
            first["cps"]["above_hard_cap_percent"],
            second["cps"]["above_hard_cap_percent"],
        ),
        (
            "Median length ratio to reference",
            first["length_ratio_to_reference"]["median"],
            second["length_ratio_to_reference"]["median"],
        ),
        (
            "90th percentile length ratio",
            first["length_ratio_to_reference"]["p90"],
            second["length_ratio_to_reference"]["p90"],
        ),
        (
            "Glossary adherence (%)",
            first["glossary"]["adherence_percent"],
            second["glossary"]["adherence_percent"],
        ),
    ):
        metrics_rows.append(
            f"<tr><th>{html.escape(name)}</th><td>{_score(left)}</td><td>{_score(right)}</td></tr>"
        )
    message = (
        "REGRESSION: candidate chrF++ is below baseline"
        if metrics["regression"]
        else "No chrF++ regression detected"
    )
    warnings = "".join(
        f"<li>{html.escape(_safe_text(str(item)))}</li>" for item in metrics.get("warnings", [])
    )
    label = (
        f'<p class="run-label"><strong>Run:</strong> '
        f"{html.escape(_safe_text(metrics['label']))}</p>"
        if metrics.get("label")
        else ""
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Subtitle model comparison</title><style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 2rem; line-height: 1.45; }} h1 {{ margin-bottom: .25rem; }}
.table-wrap {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #8886; padding: .65rem; vertical-align: top; text-align: start; }}
td[dir] {{ unicode-bidi: isolate; white-space: pre-wrap; min-width: 13rem; }}
small {{ color: #777; overflow-wrap: anywhere; }} .cps {{ display: block; margin-top: .5rem; }}
del {{ background: #b73b3b55; text-decoration: line-through; }}
ins {{ background: #1e9d6255; text-decoration: none; }}
.summary {{ max-width: 50rem; }} details {{ margin-top: .6rem; font-size: .85em; }}
.run-label {{ padding: .65rem; border-inline-start: .25rem solid #d98c24; }}
</style></head><body><h1>Subtitle model comparison</h1>
{label}
<p>{html.escape(direction)} · {len(records)} scored pairs · {len(selected)} displayed ·
{html.escape(message)}</p>
<p>Diffs compare the baseline output to the candidate, not to the reference.
Timed CPS is an aligned-group proxy.</p>
<ul>{warnings}</ul>
<table class="summary"><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th></tr></thead>
<tbody>{"".join(metrics_rows)}</tbody></table>
<h2>Human review</h2><div class="table-wrap"><table><thead><tr><th>Pair</th><th>Source</th>
<th>Reference</th><th>Baseline</th><th>Candidate</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
</body></html>"""
    return document, len(selected)
