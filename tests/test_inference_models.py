"""Actual local Transformers execution using tiny original random models."""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from ml.inference.base import TranslationRequest
from ml.inference.glossary import TransformersEntityExtractor
from ml.inference.nllb_translator import NllbTranslator
from ml.train.config import load_config
from ml.train.merge_adapter import merge_adapter
from ml.train.train_lora import run_training
from tests.test_eval_pipeline import make_evaluation_data  # noqa: F401


def test_real_local_tiny_adapter_token_scores_constraints_and_cli(
    evaluation_data, tmp_path, monkeypatch
):
    dataset, _, _ = evaluation_data
    config = replace(
        load_config(Path("ml/train/config.yaml")),
        dataset_dir=str(dataset),
        output_dir=str(tmp_path / "training"),
    )
    manifest = run_training(config, smoke_test=True)
    adapter = tmp_path / "training" / "adapter"
    kwargs = dict(
        direction="en-ckb",
        model_version_id="synthetic-v1",
        adapter_dir=adapter,
        local_files_only=True,
    )
    with pytest.raises(ValueError, match="smoke"):
        NllbTranslator(**kwargs)
    with pytest.raises(ValueError, match="smoke"):
        NllbTranslator(
            direction="en-ckb",
            model_version_id="random-base",
            model=str(tmp_path / "training" / "smoke_base"),
            local_files_only=True,
        )
    with pytest.raises(ValueError, match="immutable"):
        NllbTranslator(**kwargs, model_revision="main", allow_smoke_model=True)
    with pytest.raises(ValueError, match="recorded"):
        NllbTranslator(**kwargs, model_revision="a" * 40, allow_smoke_model=True)
    translator = NllbTranslator(**kwargs, allow_smoke_model=True)
    results = translator.translate_batch(
        [
            TranslationRequest(
                "Can you see the bridge?", context=("Start walking.",), speaker="JOHN"
            ),
            TranslationRequest("Please keep moving."),
        ]
    )
    assert len(results) == 2
    assert results[0].metadata["context_sentences_used"] == 1
    assert results[0].token_logprobs and all(
        math.isfinite(x) and x <= 0 for x in results[0].token_logprobs
    )
    # Compare streaming selected-token scores against a full teacher-forced
    # forward pass; generation is replaced only to fix the selected sequence.
    request = TranslationRequest("Take the book home.")
    encoded = translator._encoded(request, 0)
    inputs = translator.tokenizer.pad(
        [{key: encoded[key] for key in ("input_ids", "attention_mask")}], return_tensors="pt"
    )
    token_ids = translator.tokenizer.encode("کتێبەکە", add_special_tokens=False)
    sequence = torch.tensor(
        [
            [
                translator.model.config.decoder_start_token_id,
                translator.target_id,
                *token_ids,
                translator.tokenizer.eos_token_id,
            ]
        ]
    )
    with torch.inference_mode():
        logits = translator.model(
            **inputs, decoder_input_ids=sequence[:, :-1], use_cache=False
        ).logits
        expected = torch.log_softmax(logits[0, 1 : 1 + len(token_ids)].float(), dim=-1)
        expected = expected.gather(1, torch.tensor(token_ids).unsqueeze(1)).squeeze(1).tolist()
    with monkeypatch.context() as patch:
        patch.setattr(translator.model, "generate", lambda **kwargs: sequence)
        streamed = translator._generate([request], [encoded])[0]
    assert streamed.token_logprobs == pytest.approx(expected, abs=1e-5)
    constrained = translator.translate_batch(
        [TranslationRequest("Take the book home.", force_terms=("کتێبەکە",))]
    )[0]
    assert "کتێبەکە" in constrained.text and constrained.metadata["constrained"]
    with pytest.raises(ValueError, match="cannot be represented"):
        translator.translate_batch([TranslationRequest("Take it.", force_terms=("نامۆترینوشە",))])
    with pytest.raises(ValueError, match="direction"):
        NllbTranslator(**{**kwargs, "direction": "ckb-en"}, allow_smoke_model=True)
    wrong = tmp_path / "different-base"
    wrong.mkdir()
    (wrong / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        NllbTranslator(**kwargs, model=str(wrong), allow_smoke_model=True)

    merged_path = tmp_path / "merged"
    merge_adapter(adapter, merged_path, local_files_only=True)
    with pytest.raises(ValueError, match="smoke"):
        NllbTranslator(
            direction="en-ckb",
            model_version_id="merged",
            model=str(merged_path),
            local_files_only=True,
        )
    settings = manifest["config"]
    merged = NllbTranslator(
        direction="en-ckb",
        model_version_id="merged",
        model=str(merged_path),
        local_files_only=True,
        allow_smoke_model=True,
        max_source_length=settings["max_source_length"],
        max_target_length=settings["max_target_length"],
        num_beams=settings["training"]["generation_num_beams"],
    )
    merged_result = merged.translate_batch([TranslationRequest("Please keep moving.")])[0]
    assert merged_result.text == results[1].text
    assert merged_result.token_logprobs == pytest.approx(results[1].token_logprobs, abs=1e-5)

    import shutil

    import huggingface_hub.constants
    import requests
    import transformers.utils.hub

    cache = tmp_path / "hub-cache"
    storage = cache / "models--local--base"
    commit = "b" * 40
    snapshot = storage / "snapshots" / commit
    shutil.copytree(tmp_path / "training" / "smoke_base", snapshot)
    (storage / "refs").mkdir()
    (storage / "refs" / "main").write_text(commit)
    with monkeypatch.context() as patch:
        patch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
        patch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
        patch.setattr(transformers.utils.hub, "_is_offline_mode", False)

        def deny_http(*args, **kwargs):
            raise AssertionError("cached local-only model attempted a network request")

        patch.setattr(requests.sessions.Session, "request", deny_http)
        cached = NllbTranslator(
            direction="en-ckb",
            model_version_id="cached-smoke",
            model="local/base",
            local_files_only=True,
            allow_smoke_model=True,
            max_source_length=settings["max_source_length"],
            max_target_length=settings["max_target_length"],
        )
        assert cached.model_info["source"] == "local/base"
        assert cached.model_info["resolved_source"] == str(snapshot)
        assert cached.model_info["resolved_revision"] == commit
        assert len(cached.translate_batch([TranslationRequest("Please keep moving.")])) == 1

    subtitle = tmp_path / "demo.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nJOHN: Please keep\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nmoving.\n",
        encoding="utf-8",
    )
    output = tmp_path / "translated.json"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.inference.pipeline",
            "--input",
            str(subtitle),
            "--output",
            str(output),
            "--direction",
            "en-ckb",
            "--model-version-id",
            "synthetic-v1",
            "--adapter-dir",
            str(adapter),
            "--allow-smoke-model",
            "--local-files-only",
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [(cue["start_ms"], cue["end_ms"]) for cue in report["cues"]] == [
        (1000, 2000),
        (2000, 4000),
    ]
    assert report["sentences"][0]["token_logprobs"]
    assert report["model"]["warnings"] and "smoke" in report["model"]["warnings"][0]
    assert '"stage": "emit"' in completed.stderr


def test_reverse_random_adapter_uses_english_target_and_sorani_context(evaluation_data, tmp_path):
    dataset, _, _ = evaluation_data
    config = replace(
        load_config(Path("ml/train/config.yaml")),
        dataset_dir=str(dataset),
        direction="ckb-en",
        output_dir=str(tmp_path / "reverse-training"),
    )
    run_training(config, smoke_test=True)
    translator = NllbTranslator(
        direction="ckb-en",
        model_version_id="reverse-smoke",
        local_files_only=True,
        adapter_dir=tmp_path / "reverse-training" / "adapter",
        allow_smoke_model=True,
    )
    assert translator.target_id == translator.tokenizer.convert_tokens_to_ids("eng_Latn")
    result = translator.translate_batch(
        [TranslationRequest("سەیارەکە لە کوێیە؟", context=("دەست بە ڕۆیشتن بکە.",))]
    )[0]
    assert isinstance(result.text, str)
    assert result.metadata["context_sentences_used"] == 1


def test_local_only_models_never_attempt_hub_requests(monkeypatch):
    import huggingface_hub.constants
    import requests
    import transformers.utils.hub

    # Do not rely on the suite's global offline environment: local_files_only
    # must also work when the process as a whole is online.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(transformers.utils.hub, "_is_offline_mode", False)

    def deny_http(*args, **kwargs):
        raise AssertionError("local_files_only attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_http)
    with pytest.raises(ValueError, match="not available in the local Hub cache"):
        TransformersEntityExtractor("arena-local-only/nonexistent-ner", local_files_only=True)
    with pytest.raises(ValueError, match="not available in the local Hub cache"):
        NllbTranslator(
            model="arena-local-only/nonexistent-base",
            direction="en-ckb",
            model_version_id="uncached",
            local_files_only=True,
        )


def test_real_local_transformer_ner_pipeline(tmp_path, monkeypatch):
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    root = tmp_path / "ner"
    root.mkdir()
    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "Alice": 4, "met": 5, "Bob": 6, ".": 7}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=32,
    )
    tokenizer.save_pretrained(root)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        id2label={0: "O", 1: "B-PER", 2: "B-LOC", 3: "B-ORG"},
        label2id={"O": 0, "B-PER": 1, "B-LOC": 2, "B-ORG": 3},
    )
    model = BertForTokenClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))
    model.save_pretrained(root)
    extractor = TransformersEntityExtractor(str(root), local_files_only=True)
    entities = extractor.extract(["Alice met Bob.", "Alice " * 80])
    assert any(entity.text == "Alice" and entity.category == "person" for entity in entities)

    # Emulate a cached Hub snapshot and verify a Hub ID resolves to local files
    # even when the global Hugging Face offline flag is disabled.
    import shutil

    import huggingface_hub.constants
    import requests
    import transformers.utils.hub

    cache = tmp_path / "hub-cache"
    storage = cache / "models--local--ner"
    commit = "a" * 40
    snapshot = storage / "snapshots" / commit
    shutil.copytree(root, snapshot)
    (storage / "refs").mkdir()
    (storage / "refs" / "main").write_text(commit)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(transformers.utils.hub, "_is_offline_mode", False)

    def deny_http(*args, **kwargs):
        raise AssertionError("cached local-only model attempted a network request")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_http)
    cached = TransformersEntityExtractor("local/ner", local_files_only=True)
    assert cached.resolved_source == str(snapshot)
    assert cached.model_revision == commit
    assert any(entity.text == "Alice" for entity in cached.extract(["Alice met Bob."]))
