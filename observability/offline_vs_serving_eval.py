"""
offline_vs_serving_eval.py -- Stage 3: training-serving skew demonstration

THE CORE CLAIM THIS SCRIPT PROVES OR DISPROVES:
Run the EXACT SAME model, on the EXACT SAME held-out test set, through
two different preprocessing paths:
  1. "Offline eval" -- training-time preprocessing (identity, no change).
     This is what evaluate.py already does, and what a CI/CD quality
     gate would check before deployment.
  2. "Serving eval" -- production preprocessing, which has a text-
     truncation bug (see skew_scenarios.py) that offline eval has no way
     to know about, because it was never told to apply it.

If offline accuracy stays high while serving accuracy drops meaningfully,
that's the exact LinkedIn JD scenario: a model that PASSES the CI/CD
quality gate but SILENTLY degrades in production, because the gate only
ever tested the training-time code path.

If serving accuracy does NOT drop much, that's also a real, reportable
finding (same spirit as Stage 2's negative result) -- it would mean this
particular skew (60-char truncation) isn't severe enough to matter for
this model/task, and the honest move is to say so and try a shorter
truncation length, not pretend the demo worked.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import TicketClassifier, CATEGORIES  # noqa: E402
from skew_scenarios import identity, serving_truncation_skew  # noqa: E402

DATA_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "skew_results"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_with_preprocessing(classifier: TicketClassifier, test_set, preprocess_fn, tag: str):
    """Runs the test set through the classifier with a given text
    preprocessing function applied BEFORE the model sees it. Returns
    per-example results and a summary, same shape as evaluate.py's
    output so the two are directly comparable."""
    results = []
    for ex in test_set:
        preprocessed_text = preprocess_fn(ex["text"])
        predicted, confidence, latency_ms = classifier.predict(preprocessed_text)
        results.append({
            "original_text": ex["text"],
            "preprocessed_text": preprocessed_text,
            "true_label": ex["label"],
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == ex["label"],
        })

    n = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / n
    avg_confidence = sum(r["confidence"] for r in results) / n

    per_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_cat[r["true_label"]]["total"] += 1
        if r["correct"]:
            per_cat[r["true_label"]]["correct"] += 1

    print(f"\n=== {tag} ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{n})")
    print(f"Avg confidence: {avg_confidence:.3f}")
    print("Per-category accuracy:")
    for cat in CATEGORIES:
        stats = per_cat[cat]
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {cat:20s} {acc:.1%}  ({stats['correct']}/{stats['total']})")

    return results, {"accuracy": accuracy, "avg_confidence": avg_confidence,
                      "per_category": {c: (per_cat[c]["correct"] / per_cat[c]["total"] if per_cat[c]["total"] else 0)
                                        for c in CATEGORIES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=60,
                         help="Truncation length for the simulated serving-side skew bug.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    test_set = load_jsonl(DATA_DIR / "test.jsonl")

    def serving_fn(text):
        return serving_truncation_skew(text, max_chars=args.max_chars)

    classifier = TicketClassifier(rank=args.rank)

    offline_results, offline_summary = evaluate_with_preprocessing(
        classifier, test_set, identity, "OFFLINE EVAL (training-time preprocessing -- what CI/CD checks)"
    )
    serving_results, serving_summary = evaluate_with_preprocessing(
        classifier, test_set, serving_fn,
        f"SERVING EVAL (production skew: truncated to {args.max_chars} chars -- what actually ships)"
    )

    print(f"\n=== Training-serving skew summary ===")
    print(f"Offline (CI/CD gate) accuracy:  {offline_summary['accuracy']:.1%}")
    print(f"Serving (production) accuracy:  {serving_summary['accuracy']:.1%}")
    gap = offline_summary['accuracy'] - serving_summary['accuracy']
    print(f"Gap: {gap:.1%} points of accuracy invisible to offline evaluation")

    if gap > 0.05:
        print("\n>>> This IS the LinkedIn JD scenario: offline eval passes cleanly "
              "while production silently degrades, because the CI/CD gate only "
              "ever exercised the training-time preprocessing path, never the "
              "actual serving code. <<<")
    else:
        print(f"\n>>> At {args.max_chars} chars, this skew is not severe enough to "
              f"meaningfully move accuracy for this model/task -- worth trying a "
              f"shorter --max-chars value rather than reporting a weak effect as "
              f"if it were a strong one. <<<")

    with open(RESULTS_DIR / "offline_results.json", "w") as f:
        json.dump(offline_results, f, indent=2)
    with open(RESULTS_DIR / "serving_results.json", "w") as f:
        json.dump(serving_results, f, indent=2)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"max_chars": args.max_chars, "offline": offline_summary,
                    "serving": serving_summary, "gap": gap}, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
