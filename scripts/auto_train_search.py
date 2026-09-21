"""
auto_train_search.py -- automated data-clean -> train -> eval -> repeat loop

Runs a grid search over (rank, learning_rate, epochs), training and
evaluating each config in ONE continuous pod session (no per-config
pod provisioning -- that would be slow and wasteful). Stops early once
improvement stalls, so it doesn't burn GPU time on a full grid when the
answer has already stabilized.

DESIGN NOTE ON CODE DUPLICATION: this script duplicates train_lora.py's
core training logic (build_example, LoRA config, Trainer setup) rather
than importing/refactoring it. That's a deliberate tradeoff, not an
oversight -- train_lora.py is already tested and used directly by the
README's documented workflow; refactoring it to be loop-callable risked
breaking a script that's known to work, for the benefit of this one
search script. A production version would factor out a shared
train_one_config() function used by both; noting that here as a real
"what I'd do differently" rather than silently duplicating it.

USAGE: python3 scripts/auto_train_search.py
"""

import csv
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset"
SEARCH_ROOT = REPO_ROOT / "checkpoints" / "search"
RESULTS_CSV = REPO_ROOT / "checkpoints" / "search" / "search_results.csv"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CATEGORIES = [
    "Hardware", "Software", "Network", "Access_Account",
    "Password_Reset", "Email", "Security_Phishing", "Printer",
]
SYSTEM_PROMPT = (
    "You are an IT support ticket classifier. Read the ticket and respond "
    "with exactly one category from this list: "
    + ", ".join(CATEGORIES) + ". Respond with only the category name."
)

# ---- Search space ----
RANKS = [8, 16, 32]
LEARNING_RATES = [1e-4, 2e-4, 3e-4]
EPOCH_COUNTS = [2, 3, 4]
# 27 total configs, deterministic order (rank outer, then lr, then epochs).
# Order matters for early stopping -- documented here rather than hidden,
# since a different order could plausibly stop at a different point.

# ---- Early stopping ----
PATIENCE = 3          # consecutive non-improving configs before stopping
MIN_DELTA = 0.005      # an improvement smaller than this doesn't reset patience


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def data_cleaning_gate(train_raw, val_raw):
    """Validates training/validation data BEFORE any GPU time is spent --
    the same principle as Stage 4's ingestion validators, applied here to
    training data instead of production telemetry. Catches the kind of
    silent problem that would otherwise waste an entire training run:
    empty text, an invalid/misspelled label, etc.
    Returns nothing on success; raises with a clear message on failure so
    the loop fails fast rather than training on bad data.
    """
    problems = []
    for split_name, rows in [("train", train_raw), ("val", val_raw)]:
        for i, r in enumerate(rows):
            text = r.get("text", "")
            label = r.get("label", "")
            if not text or len(text.strip()) < 5:
                problems.append(f"{split_name}[{i}]: empty/too-short text")
            if label not in CATEGORIES:
                problems.append(f"{split_name}[{i}]: invalid label {label!r}")
    if problems:
        raise ValueError(
            f"Data cleaning gate FAILED with {len(problems)} problem(s), "
            f"first few: {problems[:5]}. Fix the dataset before training."
        )
    print(f"Data cleaning gate PASSED: {len(train_raw)} train, {len(val_raw)} val examples, "
          f"all texts non-empty, all labels valid.\n")


def build_example(tokenizer, text, label, max_len=512):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    label_ids = tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def train_one_config(tokenizer, train_ds, val_ds, rank, lr, epochs, output_dir):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=15,
        weight_decay=0.01, logging_steps=50, eval_strategy="epoch",
        save_strategy="no",  # search checkpoints are re-derivable; only save the FINAL winner explicitly
        bf16=True, report_to="none", disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                       eval_dataset=val_ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Clean up explicitly -- running many configs in one process will OOM
    # if GPU memory from the previous config isn't released.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_val_accuracy(tokenizer, adapter_dir, val_raw):
    """Real generation-based accuracy on the validation set -- not the
    training loss. Reuses the exact same single-forward-pass prediction
    approach as observability/model_utils.py's TicketClassifier, so the
    metric this search optimizes for is the same one the rest of the
    project already validated end to end.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    category_first_token = {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in CATEGORIES}

    correct = 0
    for ex in val_raw:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["text"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            predicted = max(category_first_token, key=lambda c: probs[category_first_token[c]].item())
        if predicted == ex["label"]:
            correct += 1

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return correct / len(val_raw)


def main():
    train_raw = load_jsonl(DATA_DIR / "train.jsonl")
    val_raw = load_jsonl(DATA_DIR / "val.jsonl")
    data_cleaning_gate(train_raw, val_raw)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = Dataset.from_list([build_example(tokenizer, r["text"], r["label"]) for r in train_raw])
    val_ds = Dataset.from_list([build_example(tokenizer, r["text"], r["label"]) for r in val_raw])

    SEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    best_acc = -1.0
    best_config = None
    stall_count = 0

    configs = [(r, lr, ep) for r in RANKS for lr in LEARNING_RATES for ep in EPOCH_COUNTS]
    print(f"Grid search: {len(configs)} total configs, patience={PATIENCE}, min_delta={MIN_DELTA}\n")

    for i, (rank, lr, epochs) in enumerate(configs):
        tag = f"rank{rank}_lr{lr}_ep{epochs}"
        output_dir = SEARCH_ROOT / tag
        print(f"[{i+1}/{len(configs)}] Training {tag} ...")

        train_one_config(tokenizer, train_ds, val_ds, rank, lr, epochs, output_dir)
        val_acc = evaluate_val_accuracy(tokenizer, output_dir / "final", val_raw)
        print(f"[{i+1}/{len(configs)}] {tag}: val_accuracy={val_acc:.1%}")

        results.append({"rank": rank, "learning_rate": lr, "epochs": epochs,
                         "val_accuracy": val_acc, "output_dir": str(output_dir)})

        if val_acc > best_acc + MIN_DELTA:
            best_acc = val_acc
            best_config = results[-1]
            stall_count = 0
            print(f"  -> new best: {best_acc:.1%}")
        else:
            stall_count += 1
            print(f"  -> no improvement ({stall_count}/{PATIENCE} stall count)")

        if stall_count >= PATIENCE:
            print(f"\nStopping early: {PATIENCE} consecutive configs without improvement "
                  f"(current best: {best_config['rank']}/{best_config['learning_rate']}/"
                  f"{best_config['epochs']} epochs at {best_acc:.1%}).")
            break

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "learning_rate", "epochs", "val_accuracy", "output_dir"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: -r["val_accuracy"]))

    print(f"\n=== Search complete: {len(results)}/{len(configs)} configs run ===")
    print(f"Best config: rank={best_config['rank']}, lr={best_config['learning_rate']}, "
          f"epochs={best_config['epochs']}, val_accuracy={best_acc:.1%}")
    print(f"Best checkpoint: {best_config['output_dir']}/final")
    print(f"Full results: {RESULTS_CSV}")
    print("\nRecommended next step: run scripts/evaluate.py or "
          "observability/offline_vs_serving_eval.py against the best checkpoint "
          "for the full rigorous evaluation (this search only computed val accuracy, "
          "not the full base-comparison/consistency/failure-analysis suite).")


if __name__ == "__main__":
    main()
