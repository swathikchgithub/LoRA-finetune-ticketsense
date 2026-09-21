"""
simulate_ingestion.py -- Stage 4: ticket ingestion with data quality issues

Extends the traffic simulation with realistic ticket METADATA (priority,
source system, requester email) -- needed because "null field" and
"schema violation" only mean something once there are structured fields
beyond raw text. Injects three distinct, separable data quality problems
at different points in the simulated 90-day period, and validates every
record with data_quality.py BEFORE deciding whether to call the model --
this ordering (validate, then maybe skip the model) is the actual point
of this stage.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import TicketClassifier, CATEGORIES  # noqa: E402
from data_quality import validate_ticket, has_blocking_issue, KNOWN_SOURCE_SYSTEMS  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "telemetry.db"
SIM_START_DATE = date(2026, 1, 1)

# Injection points, deliberately spread across the period so each issue's
# effect is separable in the resulting time series rather than confounded
# with the others:
NULL_TEXT_START_DAY = 20     # a form-submission bug starts dropping ticket text
SILENT_DEFAULT_START_DAY = 45  # an upstream integration starts silently defaulting priority
SCHEMA_VIOLATION_START_DAY = 65  # a new, unexpected source_system value starts appearing


def render_ticket(template: str, rng: random.Random) -> str:
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
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_day INTEGER NOT NULL,
            sim_date TEXT NOT NULL,
            ticket_text TEXT,
            priority TEXT,
            source_system TEXT,
            true_category TEXT,
            blocked INTEGER NOT NULL,
            issue_types TEXT NOT NULL DEFAULT '',
            predicted_category TEXT,
            confidence REAL,
            correct INTEGER,
            scenario TEXT NOT NULL DEFAULT 'quality'
        )
    """)
    conn.commit()
    return conn


def build_record(day: int, true_category: str, ticket_text: str, rng: random.Random):
    """Builds a ticket record with metadata, applying whichever quality
    issue is 'active' for this day -- issues are cumulative once started
    (a real regression doesn't fix itself), matching how a real unfixed
    upstream bug would behave over time.
    """
    priority = rng.choice(["P1", "P2", "P3", "P4", "P5"])
    source_system = rng.choice(list(KNOWN_SOURCE_SYSTEMS))
    text = ticket_text

    # Issue 1: null/empty text, affecting ~15% of records once active.
    if day >= NULL_TEXT_START_DAY and rng.random() < 0.15:
        text = "" if rng.random() < 0.5 else None

    # Issue 2: silent default priority, affecting ~30% of records once
    # active (representing one integration among several being broken).
    if day >= SILENT_DEFAULT_START_DAY and rng.random() < 0.30:
        priority = "UNSET"

    # Issue 3: schema violation, a new unexpected source_system value
    # appearing in ~20% of records once active.
    if day >= SCHEMA_VIOLATION_START_DAY and rng.random() < 0.20:
        source_system = "new_chatbot_v2"  # not in KNOWN_SOURCE_SYSTEMS

    return {"ticket_text": text, "priority": priority, "source_system": source_system}


def simulate(days: int, tickets_per_day: int, rank: int, seed: int):
    rng = random.Random(seed)
    conn = init_db(DB_PATH)
    classifier = TicketClassifier(rank=rank)

    print(f"Loaded fine-tuned model (rank={rank}). Simulating {days} days x "
          f"{tickets_per_day} tickets/day with injected data quality issues:\n"
          f"  null/empty text starting day {NULL_TEXT_START_DAY}\n"
          f"  silent default priority starting day {SILENT_DEFAULT_START_DAY}\n"
          f"  schema violation (source_system) starting day {SCHEMA_VIOLATION_START_DAY}\n")

    for day in range(days):
        sim_date = SIM_START_DATE + timedelta(days=day)
        blocked_count = 0
        issue_count = 0

        for _ in range(tickets_per_day):
            true_category = rng.choice(CATEGORIES)
            template = rng.choice(TEMPLATES[true_category])
            ticket_text = render_ticket(template, rng)

            record = build_record(day, true_category, ticket_text, rng)
            issues = validate_ticket(record)
            blocked = has_blocking_issue(issues)
            issue_types = ",".join(i["issue_type"] for i in issues)

            predicted, confidence, correct = None, None, None
            if not blocked:
                # Only call the model if the record passed validation --
                # this is the actual point of the stage: bad data gets
                # caught and skipped BEFORE it wastes a model call.
                predicted, confidence, _ = classifier.predict(record["ticket_text"])
                correct = int(predicted == true_category)
            else:
                blocked_count += 1
            if issues:
                issue_count += 1

            conn.execute(
                """INSERT INTO ingestion_log
                   (sim_day, sim_date, ticket_text, priority, source_system,
                    true_category, blocked, issue_types, predicted_category,
                    confidence, correct, scenario)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (day, sim_date.isoformat(), record["ticket_text"], record["priority"],
                 record["source_system"], true_category, int(blocked), issue_types,
                 predicted, confidence, correct, "quality"),
            )

        conn.commit()
        print(f"Day {day:3d} ({sim_date}): {issue_count}/{tickets_per_day} records "
              f"had a quality issue, {blocked_count} blocked before reaching the model")

    conn.close()
    print(f"\nDone. Ingestion log written to {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--tickets-per-day", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    simulate(args.days, args.tickets_per_day, args.rank, args.seed)
