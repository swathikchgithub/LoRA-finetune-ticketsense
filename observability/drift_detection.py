"""
drift_detection.py -- Stage 2: data drift detection

Computes Population Stability Index (PSI) and KL divergence between a
training-time baseline category distribution and each day's PRODUCTION
category distribution, using a rolling window to reduce single-day noise
(we saw ~3% std on daily accuracy from pure sampling noise in Stage 1 --
a single-day comparison would be too noisy to trust).

THE KEY STORY THIS SCRIPT IS BUILT TO DEMONSTRATE:
PSI/KL are computed on the model's PREDICTED category distribution --
which is available immediately, in real time, with zero ground truth
labels required. Accuracy, by contrast, requires knowing the TRUE label,
which in a real production system is often delayed (tickets get labeled
by a human reviewer hours/days later, if ever) or unavailable for most
traffic. This script deliberately computes both and reports the GAP
between when each one would have fired an alert -- because that gap is
the actual business case for input-distribution monitoring: it's an
early-warning signal that doesn't wait on ground truth.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "telemetry.db"
TRAIN_PATH = REPO_ROOT / "dataset" / "train.jsonl"

CATEGORIES = [
    "Hardware", "Software", "Network", "Access_Account",
    "Password_Reset", "Email", "Security_Phishing", "Printer",
]

# Standard, widely-cited PSI interpretation thresholds (used across the
# credit-risk and ML-monitoring industry, not something we invented):
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25
# Below 0.10: no significant shift. 0.10-0.25: moderate shift, worth
# watching. Above 0.25: significant shift, worth investigating/alerting.

ROLLING_WINDOW_DAYS = 7
# Chosen based directly on Stage 1's own numbers: daily accuracy had a
# 3.1% std purely from sampling noise at 20 tickets/day. A 7-day window
# (~140 tickets) smooths that noise enough to trust the signal, while
# still being short enough to catch a shift within about a week.


def load_training_baseline():
    """The reference distribution: what category mix the model was
    actually TRAINED on. This -- not day 0 of simulated traffic -- is the
    correct baseline for real drift detection, since day 0 could itself
    already be drifted if you started monitoring late."""
    labels = []
    with open(TRAIN_PATH) as f:
        for line in f:
            labels.append(json.loads(line)["label"])
    counts = pd.Series(labels).value_counts()
    dist = (counts / counts.sum()).reindex(CATEGORIES, fill_value=0)
    return dist


def compute_psi(baseline: pd.Series, current: pd.Series, epsilon=1e-4):
    """Population Stability Index between two categorical distributions
    (already normalized to proportions, same category index).
    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )
    epsilon avoids log(0)/division-by-zero for categories with 0 count in
    a short window -- a real possibility with only ~140 tickets/window
    spread across 8 categories.
    """
    baseline = baseline.clip(lower=epsilon)
    current = current.clip(lower=epsilon)
    return float(((current - baseline) * np.log(current / baseline)).sum())


def compute_kl(baseline: pd.Series, current: pd.Series, epsilon=1e-4):
    """KL divergence KL(current || baseline) via scipy.stats.entropy,
    which computes this directly when given two distributions."""
    baseline = baseline.clip(lower=epsilon)
    current = current.clip(lower=epsilon)
    return float(entropy(current, baseline))


def rolling_category_distribution(df: pd.DataFrame, day: int, window: int, column: str):
    window_df = df[(df["sim_day"] > day - window) & (df["sim_day"] <= day)]
    counts = window_df[column].value_counts()
    dist = (counts / counts.sum()).reindex(CATEGORIES, fill_value=0)
    return dist


def rolling_accuracy(df: pd.DataFrame, day: int, window: int):
    window_df = df[(df["sim_day"] > day - window) & (df["sim_day"] <= day)]
    return window_df["correct"].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="drift")
    parser.add_argument("--label-delay-days", type=int, default=5,
                         help="Simulates realistic ground-truth label latency: a human "
                              "reviewer confirms the true category N days after a ticket "
                              "comes in. The accuracy-based monitor can only see labels up "
                              "to (today - this many days); the PSI/KL monitor needs no "
                              "labels at all and sees today's predictions immediately. "
                              "This delay is what creates a genuine, realistic lead-time "
                              "advantage for distribution monitoring -- without modeling "
                              "this, both signals would just be reacting to the same window "
                              "of equally-fresh data and detect at nearly the same time, "
                              "which would misrepresent the real production advantage.")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE scenario = ?", conn, params=(args.scenario,)
    )
    conn.close()

    if df.empty:
        print(f"No rows found for scenario='{args.scenario}'. "
              f"Run: python3 observability/simulate_traffic.py --scenario {args.scenario}")
        return

    baseline_dist = load_training_baseline()
    print("Training-time baseline category distribution:")
    print(baseline_dist.round(3).to_string())
    print()

    # Establish the noise band from the FIRST rolling window's worth of
    # pure-baseline days, so our "is this a real alert" threshold for
    # accuracy is grounded in this run's own observed noise, not a
    # guessed number.
    baseline_days = sorted(df["sim_day"].unique())[:ROLLING_WINDOW_DAYS]
    baseline_period_acc = df[df["sim_day"].isin(baseline_days)]["correct"].mean()

    max_day = int(df["sim_day"].max())
    rows = []
    for day in range(ROLLING_WINDOW_DAYS, max_day + 1):
        # PSI/KL: computed on TODAY's predictions -- no labels needed, so
        # this is available in real time with zero delay.
        pred_dist = rolling_category_distribution(df, day, ROLLING_WINDOW_DAYS, "predicted_category")
        psi = compute_psi(baseline_dist, pred_dist)
        kl = compute_kl(baseline_dist, pred_dist)

        # Accuracy: can only reflect labels that have actually arrived.
        # A monitor checking on "day" only has confirmed ground truth for
        # tickets up to (day - label_delay_days) -- everything more recent
        # is still awaiting human review in this simulation of reality.
        label_available_day = day - args.label_delay_days
        if label_available_day >= ROLLING_WINDOW_DAYS:
            acc = rolling_accuracy(df, label_available_day, ROLLING_WINDOW_DAYS)
        else:
            acc = np.nan  # not enough labeled history yet to report anything

        rows.append({"day": day, "psi": psi, "kl": kl, "rolling_accuracy_as_known_on_this_day": acc})

    result_df = pd.DataFrame(rows)

    # Find the first day each signal would have fired, using thresholds
    # that don't assume knowledge of when the injected drift actually
    # started -- these are the same thresholds a real monitoring system
    # would use without knowing the "answer key."
    psi_alert_days = result_df[result_df["psi"] > PSI_MODERATE_SHIFT]["day"]
    # Accuracy alert: more than 2 std below the established baseline-period
    # accuracy -- a real statistical threshold, not "any dip." NaN rows
    # (not enough labeled history yet) are naturally excluded by this
    # comparison since NaN < anything is always False.
    acc_std = df[df["sim_day"].isin(baseline_days)].groupby("sim_day")["correct"].mean().std()
    acc_threshold = baseline_period_acc - 2 * acc_std
    acc_alert_days = result_df[result_df["rolling_accuracy_as_known_on_this_day"] < acc_threshold]["day"]

    first_psi_alert = int(psi_alert_days.iloc[0]) if len(psi_alert_days) else None
    first_acc_alert = int(acc_alert_days.iloc[0]) if len(acc_alert_days) else None

    print(f"PSI/KL computed over rolling {ROLLING_WINDOW_DAYS}-day windows.\n")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Detection summary ===")
    print(f"Baseline-period accuracy (days {baseline_days[0]}-{baseline_days[-1]}): "
          f"{baseline_period_acc:.1%} (std across those days: {acc_std:.1%})")
    print(f"Accuracy alert threshold (2 std below baseline): {acc_threshold:.1%}")
    print(f"PSI moderate-shift threshold: {PSI_MODERATE_SHIFT}")

    if first_psi_alert is not None:
        print(f"\nFirst day PSI crossed the moderate-shift threshold: day {first_psi_alert}")
    else:
        print("\nPSI never crossed the moderate-shift threshold in this run.")

    if first_acc_alert is not None:
        print(f"First day rolling accuracy crossed the alert threshold: day {first_acc_alert}")
    else:
        print("Rolling accuracy never crossed the alert threshold in this run.")

    if first_psi_alert is not None and first_acc_alert is not None:
        lead_days = first_acc_alert - first_psi_alert
        if lead_days > 0:
            print(f"\n>>> PSI-based drift detection fired {lead_days} day(s) BEFORE "
                  f"the label-delayed accuracy-based alert. <<<")
            print(f"With a {args.label_delay_days}-day label delay (realistic: a human "
                  f"reviewer confirms ground truth after the fact), PSI/KL -- which needs "
                  f"no labels at all -- surfaced this shift {lead_days} day(s) earlier than "
                  f"an accuracy-based monitor could have, purely because accuracy was still "
                  f"waiting on labeled data to arrive.")
        elif lead_days < 0:
            print(f"\n>>> The accuracy-based alert fired {-lead_days} day(s) before PSI "
                  f"crossed its threshold in this run. <<<")
            print("Worth investigating why: possibly the PSI threshold is too "
                  "conservative for this shift's magnitude, or the injected drift "
                  "affects accuracy through a channel PSI on category distribution "
                  "alone doesn't fully capture. A real system would tune the PSI "
                  "threshold using historical data rather than trust a single run.")
        else:
            print(f"\n>>> Both signals fired on the same day in this run. <<<")

    result_df.to_csv(Path(__file__).resolve().parent / f"drift_results_{args.scenario}.csv", index=False)
    print(f"\nSaved day-by-day results to observability/drift_results_{args.scenario}.csv")


if __name__ == "__main__":
    main()