"""Reproducible dataset statistics; the default OOV is train-vocabulary based."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

_TOKEN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)

    return {
        "min": ordered[0],
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 2),
    }


def dataset_statistics(
    splits: dict[str, list[dict[str, Any]]],
    *,
    diagnostics: dict[str, Any],
    gold_count: int,
    gold_requested: int,
    warnings: list[str],
) -> dict[str, Any]:
    rows = [row for split in ("train", "val", "test") for row in splits[split]]
    vocabulary = {
        language: {token for row in splits["train"] for token in _tokens(row[language])}
        for language in ("english", "sorani")
    }
    split_stats = {}
    for name, split_rows in splits.items():
        oov = {}
        for language in ("english", "sorani"):
            tokens = [token for row in split_rows for token in _tokens(row[language])]
            unknown = sum(token not in vocabulary[language] for token in tokens)
            oov[language] = {
                "token_count": len(tokens),
                "oov_count": unknown if vocabulary[language] else None,
                "rate": round(unknown / len(tokens), 6)
                if tokens and vocabulary[language]
                else None,
            }
        split_stats[name] = {
            "pair_count": len(split_rows),
            "film_count": len({row["film_id"] for row in split_rows}),
            "films": sorted({row["film_id"] for row in split_rows}),
            "provenance": dict(sorted(Counter(row["provenance"] for row in split_rows).items())),
            "token_length": {
                language: _distribution([len(_tokens(row[language])) for row in split_rows])
                for language in ("english", "sorani")
            },
            "token_oov_vs_train_vocabulary": oov,
        }
    return {
        "schema_version": 1,
        "pair_count": len(rows),
        "film_count": len({row["film_id"] for row in rows}),
        "per_film_counts": dict(sorted(Counter(row["film_id"] for row in rows).items())),
        "provenance": dict(sorted(Counter(row["provenance"] for row in rows).items())),
        "train_vocabulary_size": {language: len(vocabulary[language]) for language in vocabulary},
        "oov_definition": (
            "Case-folded regex word tokens absent from the training split vocabulary; "
            "not NLLB tokenizer unknown-token rate."
        ),
        "splits": split_stats,
        "gold_review_sample": {
            "requested": gold_requested,
            "written": gold_count,
            "source_split": "test",
            "review_status": "unverified",
            "stratification": ["film_id", "provenance"],
        },
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def markdown_report(stats: dict[str, Any]) -> str:
    """Render a compact report without including any subtitle dialogue."""
    lines = [
        "# Subtitle dataset statistics",
        "",
        f"- Retained pairs: **{stats['pair_count']}**",
        f"- Retained films: **{stats['film_count']}**",
        f"- Provenance: `{stats['provenance']}`",
        f"- Review sample: **{stats['gold_review_sample']['written']}** of "
        f"{stats['gold_review_sample']['requested']} requested, all unverified",
        "",
        "## Splits",
        "",
        "| Split | Films | Pairs | English p50 / p95 tokens | Sorani p50 / p95 tokens "
        "| EN OOV | CKB OOV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, split in stats["splits"].items():
        en = split["token_length"]["english"]
        ckb = split["token_length"]["sorani"]
        en_oov = split["token_oov_vs_train_vocabulary"]["english"]["rate"]
        ckb_oov = split["token_oov_vs_train_vocabulary"]["sorani"]["rate"]
        en_rate = "n/a" if en_oov is None else f"{en_oov:.2%}"
        ckb_rate = "n/a" if ckb_oov is None else f"{ckb_oov:.2%}"
        lines.append(
            f"| {name} | {split['film_count']} | {split['pair_count']} | "
            f"{en['p50']} / {en['p95']} | {ckb['p50']} / {ckb['p95']} | {en_rate} | {ckb_rate} |"
        )
    lines.extend(
        [
            "",
            stats["oov_definition"],
            "",
            "## Film counts",
            "",
            "| Film | Pairs |",
            "| --- | ---: |",
        ]
    )
    for film, count in stats["per_film_counts"].items():
        safe_film = film.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines.append(f"| {safe_film} | {count} |")
    lines.extend(["", "## Processing diagnostics", ""])
    for key, value in stats["diagnostics"].items():
        if key != "inputs":
            lines.append(f"- **{key}**: `{value}`")
    if stats["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in stats["warnings"])
    return "\n".join(lines) + "\n"
