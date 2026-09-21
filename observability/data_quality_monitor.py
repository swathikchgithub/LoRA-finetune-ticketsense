"""
data_quality_monitor.py -- Stage 4: monitor ingestion-time data quality

Reads the ingestion log and reports, per rolling window:
  - the rate of each specific issue type (null text, silent default,
    schema violation) -- so an on-call engineer can tell WHICH upstream
    system broke, not just that "something" looks off
  - the blocking rate -- what fraction of traffic never reached the
    model at all, because it was structurally unusable

THE KEY DISTINCTION THIS STAGE DEMONSTRATES:
Blocked records have no predicted_category, no confidence, no
correct/incorrect outcome -- they never touched the model. That's
deliberate and important: it means a spike in blocked records shows up
here, in data-quality monitoring, and NOT as a drop in model accuracy or
a shift in prediction distribution (Stages 2/3's signals). If someone
paged the on-call engineer with "model accuracy dropped," and the real
cause was a form-submission bug sending empty text, that's the wrong
team looking in the wrong place. This script's whole job is making that
distinction cheap to see.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "telemetry.db"
ROLLING_WINDOW_DAYS = 7

# Alert when an issue type's rate in a rolling window exceeds this --
# chosen as a round, defensible number for a demo; a real system would
# tune this from historical false-positive rates.
ISSUE_RATE_ALERT_THRESHOLD = 0.05  # 5% of records


def rolling_rate(df: pd.DataFrame, day: int, window: int, condition_col_check):
    window_df = df[(df["sim_day"] > day - window) & (df["sim_day"] <= day)]
    if len(window_df) == 0:
        return 0.0
    return condition_col_check(window_df).mean()


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ingestion_log", conn)
    conn.close()

    if df.empty:
        print("No rows found. Run: python3 observability/simulate_ingestion.py")
        return

    max_day = int(df["sim_day"].max())
    rows = []
    for day in range(ROLLING_WINDOW_DAYS, max_day + 1):
        window_df = df[(df["sim_day"] > day - ROLLING_WINDOW_DAYS) & (df["sim_day"] <= day)]
        n = len(window_df)

        null_text_rate = window_df["issue_types"].str.contains("null_or_empty_text").sum() / n
        silent_default_rate = window_df["issue_types"].str.contains("silent_default_priority").sum() / n
        schema_violation_rate = window_df["issue_types"].str.contains("schema_violation_source_system").sum() / n
        blocked_rate = window_df["blocked"].mean()

        # Model-side signal, computed ONLY over records that actually
        # reached the model (blocked==0) -- this is what Stage 2/3's
        # monitors would see. Comparing this against the data-quality
        # rates above is exactly how you'd tell "data problem" from
        # "model problem" in practice.
        reached_model = window_df[window_df["blocked"] == 0]
        model_accuracy = reached_model["correct"].mean() if len(reached_model) else float("nan")

        rows.append({
            "day": day,
            "null_text_rate": null_text_rate,
            "silent_default_rate": silent_default_rate,
            "schema_violation_rate": schema_violation_rate,
            "blocked_rate": blocked_rate,
            "model_accuracy_on_unblocked_traffic": model_accuracy,
        })

    result_df = pd.DataFrame(rows)
    print(f"Data quality rates over rolling {ROLLING_WINDOW_DAYS}-day windows:\n")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Alert summary ===")
    for col, label in [
        ("null_text_rate", "Null/empty ticket text"),
        ("silent_default_rate", "Silent default priority"),
        ("schema_violation_rate", "Schema violation (source_system)"),
    ]:
        alert_days = result_df[result_df[col] > ISSUE_RATE_ALERT_THRESHOLD]["day"]
        if len(alert_days):
            print(f"{label}: first crossed {ISSUE_RATE_ALERT_THRESHOLD:.0%} threshold "
                  f"on day {int(alert_days.iloc[0])}")
        else:
            print(f"{label}: never crossed the {ISSUE_RATE_ALERT_THRESHOLD:.0%} threshold")

    print(f"\n=== The core distinction: bad data vs. model problem ===")
    baseline_acc = result_df["model_accuracy_on_unblocked_traffic"].iloc[0]
    final_acc = result_df["model_accuracy_on_unblocked_traffic"].iloc[-1]
    final_blocked = result_df["blocked_rate"].iloc[-1]
    print(f"Model accuracy on UNBLOCKED traffic: {baseline_acc:.1%} (early) -> {final_acc:.1%} (late)")
    print(f"Blocked rate (never reached the model): "
          f"{result_df['blocked_rate'].iloc[0]:.1%} (early) -> {final_blocked:.1%} (late)")
    print("If model accuracy stayed roughly flat while blocked rate rose, that confirms "
          "the degradation here is a DATA problem (upstream nulls/schema issues), not a "
          "MODEL problem (the model performs the same on the traffic that actually reaches it) "
          "-- exactly the distinction that determines which team gets paged.")

    result_df.to_csv(Path(__file__).resolve().parent / "data_quality_results.csv", index=False)
    print(f"\nSaved to observability/data_quality_results.csv")


if __name__ == "__main__":
    main()
