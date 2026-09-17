"""
skew_detection.py -- Stage 3: training-serving skew detection

Stages 1-2 monitor the INPUT distribution and the model's own confidence
once predictions are already flowing. This stage asks a different
question: is the serving pipeline even feeding the model what it was
trained on? A model can be perfectly accurate offline and still degrade
in production for a reason that has nothing to do with data drift --
some step between "ticket arrives" and "text reaches the model" (a
sanitizer, a length limit, a logging wrapper) processes the text
differently than training did. This is training-serving skew, and it is
invisible to Stage 2's PSI/KL monitor, which only looks at the category
distribution of predictions -- it has no way to see that the input text
itself was mangled before it ever reached the model.

WHY THIS IS TESTABLE WITHOUT A REAL SERVING SYSTEM: model_utils.py's
TicketClassifier already exposes a `normalize_fn` hook, added specifically
for this stage (see its docstring). This script uses ONE loaded model and
runs each ticket through TWO text paths -- identity (what training saw)
and a simulated serving-side transform -- and compares the SAME model's
predictions on the SAME ticket under each path. This paired design is
deliberate: comparing two independently-sampled traffic streams would
confound skew's effect with ordinary sampling noise (Stage 1 measured
~3% std in daily accuracy from sampling alone at 20 tickets/day). Pairing
on the exact same ticket isolates the skew effect from that noise.

THE KEY STORY THIS SCRIPT IS BUILT TO DEMONSTRATE: skew doesn't always
flip the predicted label. It often shows up first as a quiet confidence
drop on tickets the model still gets right -- exactly the leading
indicator model_utils.py's docstring describes. This script reports both
the label-flip rate (skew severe enough to change the answer) and the
"silent degradation" rate (confidence drops sharply, but the label
doesn't flip) -- because a monitor that only watches accuracy would miss
the second, quieter, earlier-arriving category of problem entirely.
"""

import argparse
import random
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_dataset import TEMPLATES, APPS, NOISE_PREFIXES, NOISE_SUFFIXES, TYPO_SWAPS  # noqa: E402

from model_utils import TicketClassifier, CATEGORIES  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "telemetry.db"

# Confidence drop large enough to call "meaningfully degraded" even when
# the label didn't flip. 0.15 is a deliberately conservative cutoff --
# large enough not to be ordinary forward-pass jitter between two similar
# strings, small enough to catch real pre-flip degradation before it
# becomes a wrong answer.
SILENT_DEGRADATION_THRESHOLD = 0.15


def render_ticket(template: str, rng: random.Random) -> str:
    """Identical to simulate_traffic.py's renderer -- kept in sync
    deliberately so skew-tested tickets look like the same production
    traffic Stage 1/2 already established a baseline against."""
    text = template.format(n=rng.randint(2, 45), app=rng.choice(APPS))
    text = rng.choice(NOISE_PREFIXES) + text + rng.choice(NOISE_SUFFIXES)
    if rng.random() < 0.12:
        for a, b in TYPO_SWAPS:
            if a in text and rng.random() < 0.5:
                text = text.replace(a, b, 1)
                break
    return text


def skew_truncation(text: str) -> str:
    """Simulates a serving-side length cap that training never applied --
    e.g. a webform/API gateway silently truncating the ticket body to fit
    a fixed-width field or a logging column, upstream of the model."""
    return text[:60]


def skew_lowercase(text: str) -> str:
    """Simulates an upstream sanitizer that lowercases all input for
    (unrelated) search-indexing reasons, without anyone checking whether
    the classifier's training data was case-sensitive."""
    return text.lower()


def skew_sanitize(text: str) -> str:
    """Simulates an over-eager text sanitizer: normalizes curly
    quotes/dashes to ASCII and collapses repeated punctuation ("!!!" ->
    "!", "??" -> "?"). Individually reasonable-looking cleanup steps, none
    of which were applied to the training data -- which is exactly how
    this class of bug ships in real systems: nobody intends to introduce
    skew, each step just looked harmless in isolation."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def skew_prefix_injection(text: str) -> str:
    """Simulates a ticketing-system convention leaking into the model
    input: an upstream integration prepends a ticket ID/queue tag before
    handing text to the classifier -- something the training data (plain
    ticket bodies only) never contained."""
    ticket_id = f"TCK-{random.randint(10000, 99999)}"
    return f"[{ticket_id}] Queue: L1-Support | {text}"


SKEW_FUNCTIONS = {
    "truncation": skew_truncation,
    "lowercase": skew_lowercase,
    "sanitize": skew_sanitize,
    "prefix_injection": skew_prefix_injection,
}


def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skew_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skew_type TEXT NOT NULL,
            ticket_text TEXT NOT NULL,
            true_category TEXT NOT NULL,
            trained_predicted TEXT NOT NULL,
            trained_confidence REAL NOT NULL,
            trained_latency_ms REAL NOT NULL,
            serving_predicted TEXT NOT NULL,
            serving_confidence REAL NOT NULL,
            serving_latency_ms REAL NOT NULL,
            flipped INTEGER NOT NULL,
            silently_degraded INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn


def sample_tickets(n_tickets: int, rng: random.Random):
    """Balanced sample across categories, same generator Stage 1/2 use,
    on a fresh seed so this isn't a replay of training/test/Stage-1
    traffic."""
    tickets = []
    for i in range(n_tickets):
        true_category = CATEGORIES[i % len(CATEGORIES)]
        template = rng.choice(TEMPLATES[true_category])
        tickets.append((render_ticket(template, rng), true_category))
    rng.shuffle(tickets)
    return tickets


def run_skew_type(classifier: TicketClassifier, skew_type: str, tickets, conn):
    skew_fn = SKEW_FUNCTIONS[skew_type]
    rows = []

    for ticket_text, true_category in tickets:
        classifier.normalize_fn = lambda x: x
        trained_pred, trained_conf, trained_lat = classifier.predict(ticket_text)

        classifier.normalize_fn = skew_fn
        serving_pred, serving_conf, serving_lat = classifier.predict(ticket_text)

        flipped = int(trained_pred != serving_pred)
        silently_degraded = int(
            not flipped and (trained_conf - serving_conf) > SILENT_DEGRADATION_THRESHOLD
        )

        conn.execute(
            """INSERT INTO skew_comparisons
               (skew_type, ticket_text, true_category,
                trained_predicted, trained_confidence, trained_latency_ms,
                serving_predicted, serving_confidence, serving_latency_ms,
                flipped, silently_degraded)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (skew_type, ticket_text, true_category,
             trained_pred, trained_conf, trained_lat,
             serving_pred, serving_conf, serving_lat,
             flipped, silently_degraded),
        )

        rows.append({
            "true_category": true_category,
            "trained_predicted": trained_pred,
            "trained_confidence": trained_conf,
            "serving_predicted": serving_pred,
            "serving_confidence": serving_conf,
            "flipped": flipped,
            "silently_degraded": silently_degraded,
        })

    conn.commit()
    classifier.normalize_fn = lambda x: x
    return rows


def summarize(skew_type: str, rows: list):
    n = len(rows)
    trained_acc = sum(r["trained_predicted"] == r["true_category"] for r in rows) / n
    serving_acc = sum(r["serving_predicted"] == r["true_category"] for r in rows) / n
    flip_rate = sum(r["flipped"] for r in rows) / n
    silent_rate = sum(r["silently_degraded"] for r in rows) / n
    avg_conf_drop = sum(
        r["trained_confidence"] - r["serving_confidence"] for r in rows
    ) / n

    print(f"\n=== {skew_type} ===")
    print(f"  Trained-path accuracy:  {trained_acc:.1%}")
    print(f"  Serving-path accuracy:  {serving_acc:.1%}  "
          f"(delta: {serving_acc - trained_acc:+.1%})")
    print(f"  Label flip rate:        {flip_rate:.1%}  ({sum(r['flipped'] for r in rows)}/{n} tickets)")
    print(f"  Avg confidence drop:    {avg_conf_drop:+.3f}")
    print(f"  Silent degradation:     {silent_rate:.1%}  "
          f"(confidence dropped >{SILENT_DEGRADATION_THRESHOLD} with no label flip -- "
          f"a leading indicator an accuracy-only monitor would miss entirely)")

    return {
        "skew_type": skew_type,
        "n": n,
        "trained_accuracy": trained_acc,
        "serving_accuracy": serving_acc,
        "accuracy_delta": serving_acc - trained_acc,
        "flip_rate": flip_rate,
        "avg_confidence_drop": avg_conf_drop,
        "silent_degradation_rate": silent_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tickets", type=int, default=200,
                         help="Tickets per skew type, balanced across categories.")
    parser.add_argument("--rank", type=int, default=8,
                         help="LoRA checkpoint to use (default: rank 8, our ablation winner).")
    parser.add_argument("--seed", type=int, default=77,
                         help="Different from dataset generation (42) and Stage 1 traffic "
                              "(123) on purpose -- a fresh, independent sample.")
    parser.add_argument("--skew-types", type=str, nargs="+",
                         default=list(SKEW_FUNCTIONS.keys()), choices=list(SKEW_FUNCTIONS.keys()),
                         help="Which skew scenarios to run. Defaults to all of them.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    conn = init_db(DB_PATH)
    classifier = TicketClassifier(rank=args.rank)

    print(f"Loaded fine-tuned model (rank={args.rank}). Testing {len(args.skew_types)} "
          f"skew scenario(s) on {args.n_tickets} tickets each (paired trained-vs-serving "
          f"predictions on the identical ticket per pair).\n")

    summaries = []
    for skew_type in args.skew_types:
        tickets = sample_tickets(args.n_tickets, rng)
        rows = run_skew_type(classifier, skew_type, tickets, conn)
        summaries.append(summarize(skew_type, rows))

    conn.close()

    summaries.sort(key=lambda s: s["accuracy_delta"])
    print("\n=== Ranked by accuracy impact (worst first) ===")
    for s in summaries:
        print(f"  {s['skew_type']:<18} accuracy delta {s['accuracy_delta']:+.1%}  "
              f"flip rate {s['flip_rate']:.1%}  silent degradation {s['silent_degradation_rate']:.1%}")

    import csv
    out_path = Path(__file__).resolve().parent / "skew_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved summary to observability/skew_results.csv")
    print(f"Saved {sum(s['n'] for s in summaries)} paired comparisons to "
          f"observability/telemetry.db (table: skew_comparisons)")


if __name__ == "__main__":
    main()
