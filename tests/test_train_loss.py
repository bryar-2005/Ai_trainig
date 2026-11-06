import copy

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch
import torch.nn.functional as F
from transformers import M2M100Config, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments

from ml.train.config import LoraSettings
from ml.train.model_utils import attach_lora
from ml.train.train_lora import WeightedSeq2SeqTrainer


def tiny_model():
    config = M2M100Config(
        vocab_size=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    return M2M100ForConditionalGeneration(config)


def test_loss_weights_human_and_machine_per_example_and_ignores_padding():
    torch.manual_seed(3)
    model = tiny_model().eval()
    inputs = {
        "input_ids": torch.tensor([[4, 5, 2], [5, 6, 2]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "labels": torch.tensor([[3, 4, -100], [3, 6, 2]]),
        "sample_weight": torch.tensor([1.5, 0.5]),
    }
    loss = WeightedSeq2SeqTrainer.compute_loss(None, model, inputs)
    shifted = torch.tensor([[2, 3, 4], [2, 3, 6]])
    logits = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=shifted,
    ).logits
    token_loss = F.cross_entropy(
        logits.reshape(-1, 16), inputs["labels"].reshape(-1), reduction="none", ignore_index=-100
    ).reshape(2, 3)
    means = token_loss.sum(dim=1) / torch.tensor([2, 3])
    assert torch.allclose(loss, (means * inputs["sample_weight"]).mean())


def test_accumulated_microbatch_gradients_match_full_batch():
    torch.manual_seed(4)
    full_model = tiny_model().eval()
    micro_model = copy.deepcopy(full_model)
    batch = {
        "input_ids": torch.tensor([[4, 5, 2], [5, 6, 2], [6, 7, 2], [7, 8, 2]]),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "labels": torch.tensor([[3, 4, 2], [3, 6, 2], [3, 7, 2], [3, 8, 2]]),
        "sample_weight": torch.tensor([1.5, 0.5, 1.0, 1.5]),
    }
    WeightedSeq2SeqTrainer.compute_loss(None, full_model, batch).backward()
    for start in (0, 2):
        chunk = {key: value[start : start + 2] for key, value in batch.items()}
        (WeightedSeq2SeqTrainer.compute_loss(None, micro_model, chunk) / 2).backward()
    for full, micro in zip(full_model.parameters(), micro_model.parameters(), strict=True):
        if full.grad is not None:
            assert torch.allclose(full.grad, micro.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("sample_count", [3, 4])
def test_trainer_training_step_scales_peft_microbatches(tmp_path, sample_count):
    torch.manual_seed(7)
    full_model = attach_lora(tiny_model(), LoraSettings(r=2, alpha=4, dropout=0.0), 15)
    micro_model = copy.deepcopy(full_model)
    batch = {
        "input_ids": torch.tensor([[15, 5, 2], [15, 6, 2], [15, 7, 2], [15, 8, 2]]),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "labels": torch.tensor([[3, 4, 2], [3, 6, 2], [3, 7, 2], [3, 8, 2]]),
        "sample_weight": torch.tensor([1.5, 0.5, 1.0, 1.5]),
    }
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        gradient_accumulation_steps=2,
        report_to=[],
        use_cpu=True,
    )
    trainer = WeightedSeq2SeqTrainer(model=micro_model, args=args)
    assert trainer.model_accepts_loss_kwargs
    assert trainer.accelerator.gradient_accumulation_steps == 1
    trainer.current_gradient_accumulation_steps = 2
    batch = {key: value[:sample_count] for key, value in batch.items()}
    chunks = [{key: value[start : start + 2] for key, value in batch.items()} for start in (0, 2)]
    chunks, count = trainer.get_batch_samples(iter(chunks), 2, args.device)
    assert count == sample_count
    WeightedSeq2SeqTrainer.compute_loss(None, full_model, batch).backward()
    for chunk in chunks:
        trainer.training_step(micro_model, chunk, num_items_in_batch=count)
    for full, micro in zip(full_model.parameters(), micro_model.parameters(), strict=True):
        if full.grad is not None:
            assert torch.allclose(full.grad, micro.grad, atol=1e-6, rtol=1e-5)
