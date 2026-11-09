"""Safely restore M2M100 weights omitted as tied aliases in safetensors files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class UnmaterializedModelWeightsError(ValueError):
    """A checkpoint has weights that cannot be restored without guessing."""


_M2M100_SHARED_WEIGHT = "model.shared.weight"
_M2M100_TIED_WEIGHTS = frozenset(
    {
        _M2M100_SHARED_WEIGHT,
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
        "lm_head.weight",
    }
)


def restore_m2m100_tied_weights(
    model: Any, source: str | Path, *, revision: str | None = None
) -> bool:
    """Restore a verified shared embedding left on ``meta`` by Transformers.

    Some safetensors conversions store an M2M100 tied embedding under a decoder,
    encoder, or output-head name and record the shared name as an alias in the
    file metadata. Transformers can overwrite that loaded parameter with its
    unmaterialized ``model.shared.weight``. Restore *only* the checkpoint's
    actual tensor, and only when the metadata and architecture identify the tie.
    Never initialize missing learned weights or mutate the checkpoint on disk.

    The file must already be present locally; this function never downloads it.
    Other unexpected meta parameters or buffers are refused rather than guessed.
    Return whether a repair was needed.
    """
    meta_params = {
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.is_meta
    }
    meta_buffers = {name for name, buffer in model.named_buffers() if buffer.is_meta}
    if not meta_params and not meta_buffers:
        return False

    from transformers import M2M100ForConditionalGeneration

    if (
        not isinstance(model, M2M100ForConditionalGeneration)
        or not getattr(model.config, "tie_word_embeddings", False)
        or meta_buffers
        or not meta_params.issubset(_M2M100_TIED_WEIGHTS)
    ):
        raise UnmaterializedModelWeightsError(
            "Checkpoint has unsupported unmaterialized weights; refusing to initialize them."
        )

    from safetensors import safe_open
    from torch.nn import Parameter
    from transformers.utils.hub import cached_file

    shared = model.get_input_embeddings().weight
    # from_pretrained has already resolved and loaded this checkpoint. Use
    # the exact loaded commit when possible, and never initiate a download.
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    weight_file = cached_file(
        str(source),
        "model.safetensors",
        revision=resolved_revision,
        local_files_only=True,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if not weight_file:
        raise UnmaterializedModelWeightsError(
            "Cannot find the local safetensors file needed to restore the shared embedding."
        )
    with safe_open(weight_file, framework="pt", device="cpu") as checkpoint:
        names = _M2M100_TIED_WEIGHTS.intersection(checkpoint.keys())
        if len(names) != 1:
            raise UnmaterializedModelWeightsError(
                "Checkpoint does not have exactly one saved shared-embedding tensor."
            )
        stored_name = next(iter(names))
        metadata = checkpoint.metadata() or {}
        if (
            stored_name != _M2M100_SHARED_WEIGHT
            and metadata.get(_M2M100_SHARED_WEIGHT) != stored_name
        ):
            raise UnmaterializedModelWeightsError(
                "Checkpoint metadata does not verify its shared-embedding alias."
            )
        embedding = checkpoint.get_tensor(stored_name)
        if embedding.shape != shared.shape or embedding.dtype != shared.dtype:
            raise UnmaterializedModelWeightsError(
                "Checkpoint shared embedding has an unexpected shape or dtype."
            )
        # Even when shared is materialized, it could have been randomly
        # initialized for a missing key. Always use the verified saved tensor.
        model.get_input_embeddings().weight = Parameter(
            embedding, requires_grad=shared.requires_grad
        )

    # Retie only after a real shared tensor has been established.
    model.tie_weights()
    if any(parameter.is_meta for parameter in model.parameters()) or any(
        buffer.is_meta for buffer in model.buffers()
    ):
        raise UnmaterializedModelWeightsError(
            "Checkpoint still has unmaterialized weights after restoring its verified alias."
        )
    return True
