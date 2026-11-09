"""Restore verified tied weights without replacing learned checkpoint data."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForSeq2SeqLM,
    M2M100Config,
    M2M100ForConditionalGeneration,
    PreTrainedTokenizerFast,
)

from ml.inference.base import TranslationRequest
from ml.inference.nllb_translator import NllbTranslator
from ml.train.checkpoint import UnmaterializedModelWeightsError, restore_m2m100_tied_weights

TIED = {
    "model.shared.weight",
    "model.encoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "lm_head.weight",
}


def make_checkpoint(path: Path, keeper: str, *, metadata: bool = True):
    """Make a tiny synthetic checkpoint with the Hub converter's alias layout."""
    path.mkdir()
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "</s>": 2,
        "<s>": 3,
        "eng_Latn": 4,
        "ckb_Arab": 5,
        "Hello": 6,
        "world": 7,
        ".": 8,
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="</s>",
        bos_token="<s>",
        additional_special_tokens=["eng_Latn", "ckb_Arab"],
        model_max_length=256,
    )
    tokenizer.save_pretrained(path)
    config = M2M100Config(
        vocab_size=len(tokenizer),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    torch.manual_seed(712)
    original = M2M100ForConditionalGeneration(config).eval()
    config.save_pretrained(path)
    state = {
        name: tensor.contiguous().clone()
        for name, tensor in original.state_dict().items()
        if name not in TIED or name == keeper
    }
    saved_metadata = {"format": "pt"}
    if metadata:
        saved_metadata.update({name: keeper for name in TIED if name != keeper})
    save_file(state, path / "model.safetensors", metadata=saved_metadata)
    (path / "smoke_model.json").write_text('{"smoke_test": true}', encoding="utf-8")
    return original, tokenizer


@pytest.mark.parametrize(
    "keeper",
    ["model.decoder.embed_tokens.weight", "model.encoder.embed_tokens.weight", "lm_head.weight"],
)
def test_repair_restores_checkpoint_values_and_forward_logits(tmp_path, keeper):
    path = tmp_path / "base"
    original, _ = make_checkpoint(path, keeper)
    before = (path / "model.safetensors").read_bytes()
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True).eval()
    assert model.get_input_embeddings().weight.is_meta
    with pytest.raises(NotImplementedError, match="meta tensor"):
        model.to("cpu")

    assert restore_m2m100_tied_weights(model, path)
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(buffer.is_meta for buffer in model.buffers())
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in original.state_dict().items()
    )
    assert (path / "model.safetensors").read_bytes() == before
    assert not restore_m2m100_tied_weights(model, path)

    tokens = torch.tensor([[4, 6, 8, 2]])
    with torch.inference_mode():
        expected = original(input_ids=tokens, decoder_input_ids=tokens).logits
        actual = model(input_ids=tokens, decoder_input_ids=tokens).logits
    assert torch.equal(expected, actual)
    # Training or merging must not modify the memory-mapped checkpoint on disk.
    with torch.no_grad():
        model.get_input_embeddings().weight[0, 0].add_(1)
    assert (path / "model.safetensors").read_bytes() == before


def test_canonical_checkpoint_needs_no_repair_or_file_lookup(tmp_path):
    path = tmp_path / "base"
    original, _ = make_checkpoint(path, "model.shared.weight", metadata=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    assert not restore_m2m100_tied_weights(model, tmp_path / "not-a-model")
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in original.state_dict().items()
    )


def test_real_shared_weight_is_not_assumed_to_be_pretrained_when_alias_is_meta(tmp_path):
    path = tmp_path / "base"
    original, _ = make_checkpoint(path, "model.shared.weight", metadata=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    shared = model.get_input_embeddings().weight
    model.get_input_embeddings().weight = torch.nn.Parameter(torch.zeros_like(shared))
    model.lm_head.weight = torch.nn.Parameter(torch.empty(shared.shape, device="meta"))
    assert not model.get_input_embeddings().weight.is_meta
    assert restore_m2m100_tied_weights(model, path)
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in original.state_dict().items()
    )
    assert model.lm_head.weight is model.get_input_embeddings().weight


def test_translator_recovers_before_resizing_and_runs_offline(tmp_path, monkeypatch):
    path = tmp_path / "base"
    original, tokenizer = make_checkpoint(path, "model.decoder.embed_tokens.weight")
    embedding = original.get_input_embeddings().weight.detach().clone()
    import requests

    def deny_http(*args, **kwargs):
        raise AssertionError("local model repair attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_http)
    translator = NllbTranslator(
        direction="en-ckb",
        model_version_id="synthetic-only",
        model=str(path),
        local_files_only=True,
        allow_smoke_model=True,
        max_source_length=32,
        max_target_length=8,
        num_beams=1,
    )
    assert len(translator.tokenizer) == len(tokenizer) + 1
    assert torch.equal(translator.model.get_input_embeddings().weight[: len(tokenizer)], embedding)
    assert not any(parameter.is_meta for parameter in translator.model.parameters())
    assert translator.model.lm_head.weight is translator.model.get_input_embeddings().weight
    assert "smoke" in translator.warnings[0].lower()
    result = translator.translate_batch([TranslationRequest("Hello world.")])
    assert len(result) == 1 and isinstance(result[0].text, str)


def test_missing_alias_metadata_fails_closed(tmp_path):
    path = tmp_path / "base"
    make_checkpoint(path, "model.decoder.embed_tokens.weight", metadata=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    assert model.get_input_embeddings().weight.is_meta
    with pytest.raises(UnmaterializedModelWeightsError, match="metadata"):
        restore_m2m100_tied_weights(model, path)
    assert model.get_input_embeddings().weight.is_meta


def test_unrelated_meta_parameter_is_not_initialized(tmp_path):
    path = tmp_path / "base"
    make_checkpoint(path, "model.shared.weight")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    layer = model.model.encoder.layers[0].self_attn.q_proj
    original_shared = model.get_input_embeddings().weight.detach().clone()
    layer.weight = torch.nn.Parameter(torch.empty(layer.weight.shape, device="meta"))
    with pytest.raises(UnmaterializedModelWeightsError, match="unsupported"):
        restore_m2m100_tied_weights(model, path)
    assert layer.weight.is_meta
    assert torch.equal(model.get_input_embeddings().weight, original_shared)


@pytest.mark.parametrize("problem", ["shape", "dtype"])
def test_shared_shape_or_dtype_mismatch_is_not_initialized(tmp_path, problem):
    path = tmp_path / "base"
    make_checkpoint(path, "model.decoder.embed_tokens.weight")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    shared = model.get_input_embeddings()
    shape = (5, 16) if problem == "shape" else tuple(shared.weight.shape)
    dtype = torch.float16 if problem == "dtype" else shared.weight.dtype
    shared.weight = torch.nn.Parameter(torch.empty(shape, device="meta", dtype=dtype))
    model.tie_weights()
    with pytest.raises(UnmaterializedModelWeightsError, match="shape or dtype"):
        restore_m2m100_tied_weights(model, path)
    assert model.get_input_embeddings().weight.is_meta


def test_checkpoint_absent_after_loading_is_not_guessed(tmp_path):
    path = tmp_path / "base"
    make_checkpoint(path, "model.decoder.embed_tokens.weight")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    (path / "model.safetensors").unlink()
    with pytest.raises(UnmaterializedModelWeightsError, match="Cannot find"):
        restore_m2m100_tied_weights(model, path)
    assert model.get_input_embeddings().weight.is_meta


def test_conflicting_checkpoint_tensors_are_not_guessed(tmp_path):
    path = tmp_path / "base"
    make_checkpoint(path, "model.decoder.embed_tokens.weight")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    from safetensors.torch import load_file

    checkpoint = path / "model.safetensors"
    state = load_file(checkpoint)
    state["model.shared.weight"] = state["model.decoder.embed_tokens.weight"].clone()
    replacement = path / "replacement.safetensors"
    save_file(state, replacement, metadata={"format": "pt"})
    replacement.replace(checkpoint)
    with pytest.raises(UnmaterializedModelWeightsError, match="exactly one"):
        restore_m2m100_tied_weights(model, path)
    assert model.get_input_embeddings().weight.is_meta


def test_meta_buffer_and_untied_architecture_are_refused(tmp_path):
    path = tmp_path / "base"
    make_checkpoint(path, "model.decoder.embed_tokens.weight")
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
    model.register_buffer("unexpected_meta", torch.empty(1, device="meta"))
    with pytest.raises(UnmaterializedModelWeightsError, match="unsupported"):
        restore_m2m100_tied_weights(model, path)
    del model.unexpected_meta
    model.config.tie_word_embeddings = False
    with pytest.raises(UnmaterializedModelWeightsError, match="unsupported"):
        restore_m2m100_tied_weights(model, path)
    assert model.get_input_embeddings().weight.is_meta
