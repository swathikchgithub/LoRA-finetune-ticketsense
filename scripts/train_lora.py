"""
train_lora.py

LoRA fine-tunes a small instruction model to classify IT support tickets
into one of 8 categories. Run this on a rented GPU (RunPod A10/A40,
Lambda, or Colab Pro — a single 16-24GB GPU is plenty for a 1.5B-3B model
at this LoRA rank).

Base model: Qwen2.5-1.5B-Instruct (Apache 2.0, strong instruction
following for its size, cheap to fine-tune and serve). Swap
MODEL_NAME for meta-llama/Llama-3.2-3B-Instruct if you want a larger
comparison point — same script works, just needs more VRAM.

HOW TO FRAME THE TASK:
This is a classification problem, but we frame it as constrained text
generation (the model outputs the category string) rather than adding a
classification head. Why: it's the standard, more interview-relevant
pattern for instruction-tuned LLMs, it lets us reuse the model's existing
instruction-following ability, and it generalizes to multi-label /
structured-output variants later (e.g. category + priority) without
re-architecting.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "lora-it-ticket-classifier"

CATEGORIES = [
    "Hardware", "Software", "Network", "Access_Account",
    "Password_Reset", "Email", "Security_Phishing", "Printer",
]

SYSTEM_PROMPT = (
    "You are an IT support ticket classifier. Read the ticket and respond "
    "with exactly one category from this list: "
    + ", ".join(CATEGORIES) + ". Respond with only the category name."
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_example(tokenizer, text, label, max_len=512):
    """Builds a chat-formatted training example and masks the loss so the
    model is only trained to predict the label tokens, not the prompt.
    This matters: without masking, the model wastes capacity learning to
    reproduce the (fixed) system prompt and ticket text, which slows
    convergence and can hurt generalization."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    label_ids = tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]

    input_ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids  # -100 = ignored in loss

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_jsonl(DATA_DIR / "train.jsonl")
    val_raw = load_jsonl(DATA_DIR / "val.jsonl")

    train_ds = Dataset.from_list(
        [build_example(tokenizer, r["text"], r["label"]) for r in train_raw]
    )
    val_ds = Dataset.from_list(
        [build_example(tokenizer, r["text"], r["label"]) for r in val_raw]
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ---- LoRA config: every choice explained ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        # Rank 16: for a well-defined, low-diversity task like 8-way
        # classification, the "delta" the model needs to learn is small —
        # mostly re-weighting which tokens to attend to for category
        # signal. Rank 8-16 is typically enough; going to 64+ risks
        # overfitting on ~500 training examples and slows training for
        # no accuracy gain. If eval shows underfitting (loss plateaus
        # high), raise to 32 before doing anything else.
        lora_alpha=32,
        # Alpha = 2x rank is a well-tested default (effective scaling
        # factor alpha/r = 2). It controls how strongly the LoRA update
        # is weighted relative to the frozen base weights at inference.
        lora_dropout=0.05,
        # Small dropout on the LoRA layers only, as regularization against
        # overfitting given our tiny dataset.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Attention projections are the minimum; adding the MLP
        # projections (gate/up/down) gives LoRA more capacity to shift
        # the model's "vocabulary" toward our label tokens, which matters
        # more for classification-as-generation than for style transfer.
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expect roughly 0.5-1.5% of total params trainable. If this number is
    # much higher, target_modules is too broad; much lower, LoRA may be
    # under-parameterized for the task.

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        # 3 epochs over ~500 examples. More than that on a dataset this
        # small risks memorizing surface phrasing rather than the
        # underlying category signal — watch val loss each epoch to
        # confirm it isn't rising while train loss keeps falling
        # (the standard overfitting signature).
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        # Effective batch size 16. Small effective batches add useful
        # gradient noise on a small dataset; too large and you converge
        # to a sharper, less-generalizing minimum on so few examples.
        learning_rate=2e-4,
        # LoRA typically wants a learning rate 10-100x higher than full
        # fine-tuning (2e-5 to 5e-5 range) because we're updating a much
        # smaller set of low-rank parameters from a random init, not
        # nudging pretrained weights.
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        report_to="none",  # switch to "wandb" or "tensorboard" to monitor loss curves live
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"Saved LoRA adapter to {OUTPUT_DIR / 'final'}")

    # ---- What to watch during training ----
    # 1. Train loss should fall smoothly and monotonically-ish. Spiky loss
    #    usually means learning rate is too high.
    # 2. Eval (val) loss should track train loss downward. If val loss
    #    starts rising while train loss keeps falling -> overfitting.
    #    With load_best_model_at_end=True, the checkpoint with lowest
    #    eval_loss is kept automatically.
    # 3. If both train and val loss plateau early and high -> underfitting.
    #    Try raising r to 32 or lowering dropout before adding more data.


if __name__ == "__main__":
    main()
