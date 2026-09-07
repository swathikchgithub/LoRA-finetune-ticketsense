"""
evaluate.py

This is the part of the project that should look like it came from
someone with real eval infrastructure experience, not a tutorial.

Compares BASE model (zero-shot / prompted) vs LORA FINE-TUNED model on:
  1. Accuracy overall and per-category (confusion matrix)
  2. Consistency: run each test example N times at temperature>0 and
     measure how often the model agrees with itself (a fine-tuned model
     collapsing onto the label distribution should be far more
     consistent than a prompted base model hedging between plausible
     categories)
  3. Latency: p50/p95 generation time per example
  4. Cost proxy: tokens generated per example (fine-tuned model should
     emit ~1-3 tokens; base model prompted to "explain your reasoning
     then answer" emits far more unless heavily constrained)
  5. Failure mode analysis: which categories get confused with which,
     and manual inspection of the worst N errors

Run after train_lora.py has produced checkpoints/lora-it-ticket-classifier/final
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "lora-it-ticket-classifier" / "final"
DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CATEGORIES = [
    "Hardware", "Software", "Network", "Access_Account",
    "Password_Reset", "Email", "Security_Phishing", "Printer",
]

SYSTEM_PROMPT = (
    "You are an IT support ticket classifier. Read the ticket and respond "
    "with exactly one category from this list: "
    + ", ".join(CATEGORIES) + ". Respond with only the category name."
)

CONSISTENCY_RUNS = 5  # how many times to re-sample each test example


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    # NOTE: base_model and ft_model share weights until adapters are
    # toggled — use ft_model.disable_adapter() context to get true base
    # behavior from the SAME loaded weights (avoids loading the model twice).
    return tokenizer, base_model, ft_model


def generate(model, tokenizer, text, temperature=0.0, max_new_tokens=12):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

  

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - start

    gen_tokens = out[0][inputs.shape[1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return response, elapsed, len(gen_tokens)


def normalize_prediction(raw: str) -> str:
    """Base model won't always emit a clean category string. Map its
    free-text response back to the closest known category, or 'INVALID'
    if nothing matches — this itself is a key eval signal (how often
    does the base model fail to follow the output format at all)."""
    raw_lower = raw.lower()
    for cat in CATEGORIES:
        if cat.lower().replace("_", " ") in raw_lower or cat.lower() in raw_lower:
            return cat
    return "INVALID"


def evaluate_model(model, tokenizer, test_set, temperature=0.0, tag="model"):
    results = []
    for ex in test_set:
        raw, elapsed, n_tokens = generate(model, tokenizer, ex["text"], temperature=temperature)
        pred = normalize_prediction(raw)
        results.append({
            "text": ex["text"],
            "true_label": ex["label"],
            "raw_output": raw,
            "predicted": pred,
            "correct": pred == ex["label"],
            "latency_s": elapsed,
            "output_tokens": n_tokens,
        })
    return results


def summarize(results, tag):
    n = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / n

    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_cat[r["true_label"]]["total"] += 1
        if r["correct"]:
            per_cat[r["true_label"]]["correct"] += 1

    confusion = Counter((r["true_label"], r["predicted"]) for r in results)
    latencies = sorted(r["latency_s"] for r in results)
    p50 = latencies[int(n * 0.5)]
    p95 = latencies[min(int(n * 0.95), n - 1)]
    avg_tokens = sum(r["output_tokens"] for r in results) / n
    invalid_rate = sum(1 for r in results if r["predicted"] == "INVALID") / n

    print(f"\n=== {tag} ===")
    print(f"Overall accuracy: {accuracy:.1%}  ({correct}/{n})")
    print(f"Invalid/unparseable outputs: {invalid_rate:.1%}")
    print(f"Latency p50: {p50*1000:.0f}ms | p95: {p95*1000:.0f}ms")
    print(f"Avg output tokens per prediction: {avg_tokens:.1f}")
    print("Per-category accuracy:")
    for cat in CATEGORIES:
        stats = per_cat[cat]
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {cat:20s} {acc:.1%}  ({stats['correct']}/{stats['total']})")

    print("Top confusions (true -> predicted, count):")
    confusions_only = {k: v for k, v in confusion.items() if k[0] != k[1]}
    for (true, pred), count in sorted(confusions_only.items(), key=lambda x: -x[1])[:5]:
        print(f"  {true} -> {pred}: {count}")

    return {
        "accuracy": accuracy, "invalid_rate": invalid_rate,
        "p50_ms": p50 * 1000, "p95_ms": p95 * 1000,
        "avg_output_tokens": avg_tokens,
        "per_category_accuracy": {c: (per_cat[c]["correct"] / per_cat[c]["total"] if per_cat[c]["total"] else 0) for c in CATEGORIES},
    }


def consistency_check(model, tokenizer, test_set, temperature=0.7, n_runs=CONSISTENCY_RUNS):
    """Runs each example n_runs times at temperature>0 and measures the
    fraction where all runs agree on the same prediction. A model that
    has genuinely learned the task should be far more self-consistent
    than one relying on the base model's prior + prompt alone."""
    agree_count = 0
    for ex in test_set:
        preds = []
        for _ in range(n_runs):
            raw, _, _ = generate(model, tokenizer, ex["text"], temperature=temperature)
            preds.append(normalize_prediction(raw))
        if len(set(preds)) == 1:
            agree_count += 1
    return agree_count / len(test_set)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    test_set = load_jsonl(DATA_DIR / "test.jsonl")

    tokenizer, base_model, ft_model = load_models()

    print("Evaluating BASE model (zero-shot, prompted)...")
    with ft_model.disable_adapter():
        base_results = evaluate_model(base_model, tokenizer, test_set, temperature=0.0, tag="base")
    base_summary = summarize(base_results, "BASE MODEL (zero-shot prompted)")

    print("\nEvaluating FINE-TUNED model...")
    ft_results = evaluate_model(ft_model, tokenizer, test_set, temperature=0.0, tag="finetuned")
    ft_summary = summarize(ft_results, "LORA FINE-TUNED MODEL")

    print("\nRunning consistency check (this takes a while: N runs x test set)...")
    with ft_model.disable_adapter():
        base_consistency = consistency_check(base_model, tokenizer, test_set[:30])  # subsample for cost
    ft_consistency = consistency_check(ft_model, tokenizer, test_set[:30])
    print(f"Base model self-consistency (30 examples, {CONSISTENCY_RUNS} runs each): {base_consistency:.1%}")
    print(f"Fine-tuned model self-consistency: {ft_consistency:.1%}")

    # Save everything for the README / failure analysis
    with open(RESULTS_DIR / "base_results.json", "w") as f:
        json.dump(base_results, f, indent=2)
    with open(RESULTS_DIR / "ft_results.json", "w") as f:
        json.dump(ft_results, f, indent=2)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({
            "base": base_summary, "finetuned": ft_summary,
            "base_consistency": base_consistency, "ft_consistency": ft_consistency,
        }, f, indent=2)

    print(f"\nResults written to {RESULTS_DIR}")
    print("Next: pull the worst-N errors from ft_results.json where correct=False "
          "for the failure mode section of the README.")


if __name__ == "__main__":
    main()