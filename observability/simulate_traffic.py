"""
simulate_traffic.py -- Stage 1: simulate production traffic

Generates a stream of ITSM tickets representing "production traffic" over
a simulated period, runs each through the fine-tuned model, and logs
inputs, predictions, confidence, and latency to a SQLite database that
supports time-series analysis (Stage 2+ will query this by day).

WHY SQLITE: no external dependencies (stdlib), a real file on disk you can
inspect with any SQL client, and pandas can read it directly via
pd.read_sql -- which is exactly what drift detection (Stage 2) needs for
day-over-day distribution comparisons.

This script deliberately reuses the exact same TEMPLATES dictionary from
generate_dataset.py (the training data generator) as its "ground truth"
category assigner. Two reasons: (1) it keeps "true label" meaningful for
computing simulated production accuracy, since we know what category each
generated ticket actually belongs to, and (2) later stages can shift the
category MIX away from training-time proportions to simulate drift,
without needing new templates -- the drift is in the DISTRIBUTION of
which categories appear, not in inventing new ticket types.
"""

import argparse
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_dataset import TEMPLATES, APPS, NOISE_PREFIXES, NOISE_SUFFIXES, TYPO_SWAPS  # noqa: E402

from model_utils import TicketClassifier, CATEGORIES  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "telemetry.db"
SIM_START_DATE = date(2026, 1, 1)


def render_ticket(template: str, rng: random.Random) -> str:
    """Same rendering logic as generate_dataset.py's render(), but using a
    LOCAL random.Random instance instead of the global `random` module.
    This keeps this script's randomness independent and reproducible on
    its own seed, without depending on or disturbing generate_dataset.py's
    global random state."""
    text = template.format(n=rng.randint(2, 45), app=rng.choice(APPS))
    text = rng.choice(NOISE_PREFIXES) + text + rng.choice(NOISE_SUFFIXES)
    if rng.random() < 0.12:
        for a, b in TYPO_SWAPS:
            if a in text and rng.random() < 0.5:
                text = text.replace(a, b, 1)
                break
    return text


def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_day INTEGER NOT NULL,
            sim_date TEXT NOT NULL,
            ticket_text TEXT NOT NULL,
            true_category TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            latency_ms REAL NOT NULL,
            correct INTEGER NOT NULL,
            scenario TEXT NOT NULL DEFAULT 'baseline'
        )
    """)
    conn.commit()
    return conn


def category_weights_for_day(day: int, scenario: str, drift_start_day: int):
    """Returns a dict of {category: weight} to sample from for this day.

    Baseline: equal weight across all 8 categories, matching training
    distribution.

    'drift' scenario (from drift_start_day onward): shifts weight toward
    Software and Access_Account specifically -- not an arbitrary choice.
    These are the two categories our rank-8 model handles WORST on the
    real held-out test set (77.8% and 25% accuracy respectively, see
    README) -- a real production shift that increases the share of
    tickets in your model's already-weak categories is exactly the kind
    of drift that causes visible performance degradation, as opposed to a
    shift toward categories the model already handles perfectly (which
    would show up in PSI but NOT hurt accuracy -- a distinction worth
    being able to explain: not all input drift degrades performance, but
    this specific shift plausibly represents something realistic, like an
    SSO rollout causing a wave of access and login-related tickets.
    """
    baseline_weights = {c: 1.0 for c in CATEGORIES}
    if scenario == "drift" and day >= drift_start_day:
        weights = dict(baseline_weights)
        weights["Software"] = 4.0
        weights["Access_Account"] = 4.0
        return weights
    return baseline_weights


def simulate(days: int, tickets_per_day: int, rank: int, seed: int,
             scenario: str, drift_start_day: int):
    rng = random.Random(seed)
    conn = init_db(DB_PATH)
    classifier = TicketClassifier(rank=rank)

    print(f"Loaded fine-tuned model (rank={rank}). Simulating {days} days "
          f"x {tickets_per_day} tickets/day = {days * tickets_per_day} total predictions. "
          f"Scenario={scenario}"
          + (f" (drift starts day {drift_start_day})" if scenario == "drift" else "") + "\n")

    for day in range(days):
        sim_date = SIM_START_DATE + timedelta(days=day)
        day_correct = 0
        day_confidences = []
        day_latencies = []

        weights_today = category_weights_for_day(day, scenario, drift_start_day)
        cats_today = list(weights_today.keys())
        wts_today = list(weights_today.values())

        for _ in range(tickets_per_day):
            true_category = rng.choices(cats_today, weights=wts_today, k=1)[0]
            template = rng.choice(TEMPLATES[true_category])
            ticket_text = render_ticket(template, rng)

            predicted, confidence, latency_ms = classifier.predict(ticket_text)
            correct = int(predicted == true_category)

            conn.execute(
                """INSERT INTO predictions
                   (sim_day, sim_date, ticket_text, true_category,
                    predicted_category, confidence, latency_ms, correct, scenario)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (day, sim_date.isoformat(), ticket_text, true_category,
                 predicted, confidence, latency_ms, correct, scenario),
            )

            day_correct += correct
            day_confidences.append(confidence)
            day_latencies.append(latency_ms)

        conn.commit()

        day_accuracy = day_correct / tickets_per_day
        avg_conf = sum(day_confidences) / len(day_confidences)
        avg_lat = sum(day_latencies) / len(day_latencies)
        print(f"Day {day:3d} ({sim_date}): accuracy={day_accuracy:.1%}  "
              f"avg_confidence={avg_conf:.3f}  avg_latency={avg_lat:.1f}ms")

    conn.close()
    print(f"\nDone. Telemetry written to {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--tickets-per-day", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8,
                         help="LoRA checkpoint to use (default: rank 8, our ablation winner)")
    parser.add_argument("--seed", type=int, default=123,
                         help="Different from dataset generation's seed (42) on purpose -- "
                              "this is a DIFFERENT random stream simulating new, unseen traffic, "
                              "not a replay of the training/test data.")
    parser.add_argument("--scenario", type=str, default="baseline", choices=["baseline", "drift"],
                         help="'baseline' = stable category mix throughout. 'drift' = mix "
                              "shifts starting at --drift-start-day, weighted toward the "
                              "model's weaker categories (Software, Access_Account).")
    parser.add_argument("--drift-start-day", type=int, default=45,
                         help="Day the category mix shift begins, for --scenario drift.")
    args = parser.parse_args()

    simulate(args.days, args.tickets_per_day, args.rank, args.seed, args.scenario, args.drift_start_day)