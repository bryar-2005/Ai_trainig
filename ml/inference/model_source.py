"""Resolve cached model snapshots without allowing unintended Hub requests."""

from __future__ import annotations

import re
from pathlib import Path


def resolve_model_source(
    source: str, *, revision: str | None = None, local_files_only: bool = False
) -> tuple[str, str | None]:
    """Return a loadable source and any locally resolved immutable revision.

    Some Transformers AutoModel versions inspect PEFT adapter metadata without
    forwarding ``local_files_only``. Resolving a cached Hub snapshot to its
    local directory first keeps all subsequent model/tokenizer loads offline.
    The original source should still be retained separately for provenance.
    """
    if not local_files_only or Path(source).expanduser().is_dir():
        return source, None
    from huggingface_hub import snapshot_download

    try:
        resolved = snapshot_download(repo_id=source, revision=revision, local_files_only=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Model {source!r} is not available in the local Hub cache") from exc
    commit = Path(resolved).name
    return resolved, commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None
