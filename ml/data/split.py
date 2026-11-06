"""Leakage-resistant film-level splits and a stratified human-review sample."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def split_by_film(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    allow_empty_splits: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Assign each film to exactly one split, before distributing its rows."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    films = sorted({row["film_id"] for row in rows})
    if not films:
        raise ValueError("No retained films are available to split")
    if len(films) < 3 and not allow_empty_splits:
        raise ValueError(
            f"Only {len(films)} films remain after filtering/deduplication; at least 3 are "
            "required for nonempty film-disjoint train, validation, and test splits. "
            "Add more films, or use --allow-empty-splits for development only."
        )
    random.Random(seed).shuffle(films)
    n = len(films)
    if n < 3:
        n_test = int(n == 2)
        n_val = 0
    else:
        n_val = max(1, math.floor(n * val_fraction + 0.5))
        n_test = max(1, math.floor(n * test_fraction + 0.5))
        while n_val + n_test > n - 1:
            if n_val > 1 and (n_val / val_fraction >= n_test / test_fraction or n_test == 1):
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError("Cannot allocate a nonempty training split")
    test_films = set(films[:n_test])
    val_films = set(films[n_test : n_test + n_val])
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = (
            "test"
            if row["film_id"] in test_films
            else "val"
            if row["film_id"] in val_films
            else "train"
        )
        result[split].append(row)
    for split_rows in result.values():
        split_rows.sort(key=lambda row: (row["film_id"], row["english_start_ms"], row["pair_id"]))
    return result


def sample_gold_for_review(
    test_rows: list[dict[str, Any]], *, size: int = 1000, seed: int = 42
) -> list[dict[str, Any]]:
    """Sample only the held-out test split by (film, provenance).

    Every selected row is explicitly marked unverified. A small test set is not
    duplicated or padded just to reach the requested sample size.
    """
    if size < 0:
        raise ValueError("Gold sample size must be non-negative")
    if not size or not test_rows:
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        groups[(row["film_id"], row["provenance"])].append(row)
    keys = sorted(groups)
    count = min(size, len(test_rows))
    quotas = {key: int(count >= len(keys)) for key in keys}
    remaining = count - sum(quotas.values())
    capacity = {key: len(groups[key]) - quotas[key] for key in keys}
    total_capacity = sum(capacity.values())
    remainders: list[tuple[float, tuple[str, str]]] = []
    if total_capacity:
        for key in keys:
            exact = remaining * capacity[key] / total_capacity
            whole = math.floor(exact)
            quotas[key] += whole
            remainders.append((exact - whole, key))
        leftover = count - sum(quotas.values())
        for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if not leftover:
                break
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                leftover -= 1
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in keys:
        bucket = sorted(groups[key], key=lambda row: row["pair_id"])
        selected.extend(
            {**row, "review_status": "unverified"} for row in rng.sample(bucket, quotas[key])
        )
    return sorted(
        selected, key=lambda row: (row["film_id"], row["english_start_ms"], row["pair_id"])
    )
