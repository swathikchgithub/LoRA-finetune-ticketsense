"""
alerting.py -- Stage 5: alert correlation and prioritization

THE PROBLEM THIS SOLVES: Stages 2-4 each produce a raw stream of
threshold crossings -- one alert per day, per metric, for as long as a
condition persists. Paged naively, that's dozens of pages for what's
often ONE underlying incident. This module does what an ITSM event
correlation layer does: group raw events into incidents, confirm actual
business impact before escalating, and rank by real impact rather than
statistical magnitude alone -- because a scary-looking number with zero
downstream effect and a boring-looking number that's silently losing a
fifth of your traffic deserve very different responses.

PRIORITY SCALE: reuses the P1-P5 convention already established in
data_quality.py's ticket priority field, deliberately -- same scale, same
mental model, one less thing for an on-call engineer to context-switch
between.
  P1: confirmed traffic/accuracy impact, large and sustained
  P2: confirmed impact, moderate or short-lived
  P3: no confirmed impact yet, but a persistent, growing pattern worth investigating
  P4: no confirmed impact, isolated/short-lived -- worth logging, not paging
  P5: informational / static audit findings (e.g. Stage 3's skew risk, which is
      a standing finding rather than a live time-series alert)
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

OBS_DIR = Path(__file__).resolve().parent
DRIFT_ACC_ALERT_THRESHOLD_STD = 2  # from drift_detection.py's own methodology
PSI_THRESHOLD = 0.25
DATA_QUALITY_RATE_THRESHOLD = 0.05


@dataclass
class Incident:
    source_stage: str
    alert_type: str
    start_day: int
    end_day: int
    peak_value: float
    business_impact_confirmed: bool
    impact_description: str
    priority: str = ""
    reasoning: str = ""
    blocking: bool = False


def correlate_consecutive_days(alert_days: list) -> list:
    """Groups a list of individual alert-day numbers into (start, end)
    incident spans wherever the days are consecutive -- this is the
    actual 'event correlation' step: 40 consecutive daily threshold
    breaches become ONE incident spanning those 40 days, not 40 pages.
    """
    if not alert_days:
        return []
    alert_days = sorted(alert_days)
    spans = []
    start = prev = alert_days[0]
    for d in alert_days[1:]:
        if d == prev + 1:
            prev = d
        else:
            spans.append((start, prev))
            start = prev = d
    spans.append((start, prev))
    return spans


def analyze_drift(drift_csv: Path):
    """Reads drift_detection.py's saved output and produces correlated,
    impact-confirmed incidents rather than a raw list of daily alerts."""
    if not drift_csv.exists():
        return []
    df = pd.read_csv(drift_csv)

    psi_alert_days = df[df["psi"] > PSI_THRESHOLD]["day"].tolist()
    spans = correlate_consecutive_days(psi_alert_days)

    # Establish the accuracy noise band from the data itself, same
    # methodology as drift_detection.py, so impact confirmation uses a
    # real statistical threshold rather than "any dip."
    acc_col = "rolling_accuracy_as_known_on_this_day" if "rolling_accuracy_as_known_on_this_day" in df.columns else "rolling_accuracy"
    early = df[acc_col].dropna().iloc[:7]
    baseline_acc = early.mean() if len(early) else float("nan")
    acc_std = early.std() if len(early) else 0
    impact_threshold = baseline_acc - DRIFT_ACC_ALERT_THRESHOLD_STD * acc_std

    incidents = []
    for start, end in spans:
        window = df[(df["day"] >= start) & (df["day"] <= end)]
        peak_psi = window["psi"].max()
        min_acc_in_window = window[acc_col].min()
        impact_confirmed = bool(min_acc_in_window < impact_threshold) if not pd.isna(min_acc_in_window) else False

        incidents.append(Incident(
            source_stage="Stage 2: Data Drift",
            alert_type="PSI distribution shift",
            start_day=int(start), end_day=int(end), peak_value=float(peak_psi),
            business_impact_confirmed=impact_confirmed,
            impact_description=(
                f"Rolling accuracy dropped to {min_acc_in_window:.1%} during this window "
                f"(below the {impact_threshold:.1%} impact threshold)."
                if impact_confirmed else
                f"Peak PSI={peak_psi:.2f} (well past the {PSI_THRESHOLD} significance threshold), "
                f"but rolling accuracy stayed within normal range throughout "
                f"(min {min_acc_in_window:.1%} vs. {impact_threshold:.1%} threshold) -- "
                f"this is a confirmed distribution shift with NO observed accuracy impact."
            ),
        ))
    return incidents


def analyze_skew(skew_summary_json: Path):
    """Stage 3 wasn't a live rolling monitor -- it's a standing audit
    finding (measured accuracy gap between training and serving
    preprocessing paths). Represented as a single informational/high-risk
    finding rather than a day-by-day alert, since the underlying
    condition (if unfixed) is a constant, ongoing risk, not an event with
    a start/end day."""
    if not skew_summary_json.exists():
        return []
    with open(skew_summary_json) as f:
        summary = json.load(f)

    gap = summary.get("gap", 0)
    return [Incident(
        source_stage="Stage 3: Training-Serving Skew",
        alert_type="Offline/serving accuracy gap (standing audit finding)",
        start_day=0, end_day=0, peak_value=gap,
        business_impact_confirmed=True,  # this IS a measured accuracy gap, not a hypothesis
        impact_description=(
            f"Offline (CI/CD) accuracy {summary['offline']['accuracy']:.1%} vs. serving "
            f"accuracy {summary['serving']['accuracy']:.1%} -- a {gap:.1%}-point gap "
            f"invisible to standard offline evaluation. Confirmed root cause: text "
            f"truncation divergence between training and serving preprocessing."
        ),
    )]


def analyze_data_quality(dq_csv: Path):
    if not dq_csv.exists():
        return []
    df = pd.read_csv(dq_csv)

    incidents = []
    for col, label, is_blocking in [
        ("null_text_rate", "Null/empty ticket text", True),
        ("silent_default_rate", "Silent default priority field", False),
        ("schema_violation_rate", "Schema violation (source_system)", False),
    ]:
        alert_days = df[df[col] > DATA_QUALITY_RATE_THRESHOLD]["day"].tolist()
        spans = correlate_consecutive_days(alert_days)
        for start, end in spans:
            window = df[(df["day"] >= start) & (df["day"] <= end)]
            peak_rate = window[col].max()
            if is_blocking:
                impact_text = (
                    f"Peak rate {peak_rate:.1%} of traffic affected during this window -- "
                    f"this traffic is BLOCKED before reaching the model entirely (lost, "
                    f"not just degraded)."
                )
            else:
                impact_text = (
                    f"Peak rate {peak_rate:.1%} of traffic affected during this window -- "
                    f"NON-BLOCKING: these tickets still reach the model and get classified, "
                    f"but with a corrupted/missing metadata field. The classification itself "
                    f"may be fine; downstream systems relying on this field will not be."
                )
            incidents.append(Incident(
                source_stage="Stage 4: Data Quality",
                alert_type=label,
                start_day=int(start), end_day=int(end), peak_value=float(peak_rate),
                business_impact_confirmed=True,
                impact_description=impact_text,
                blocking=is_blocking,
            ))
    return incidents


def score_priority(incident: Incident) -> Incident:
    """The actual prioritization logic: impact confirmation first,
    magnitude and duration second. An unconfirmed statistical anomaly
    never outranks a confirmed real-world impact, no matter how extreme
    its raw number looks."""
    duration = incident.end_day - incident.start_day + 1

    if not incident.business_impact_confirmed:
        # This branch is exactly Stage 2's own drift finding: a large,
        # sustained, threshold-crossing signal with no confirmed impact.
        # It stays LOW priority by design, however dramatic peak_value is.
        incident.priority = "P4"
        incident.reasoning = (
            "No confirmed business impact -- statistically significant but not "
            "yet affecting outcomes. Log and watch, don't page."
        )
        return incident

    if incident.source_stage.startswith("Stage 3"):
        incident.priority = "P2" if incident.peak_value < 0.15 else "P1"
        incident.reasoning = (
            f"Standing, confirmed accuracy gap of {incident.peak_value:.1%} between "
            f"training and serving -- ongoing risk until the preprocessing mismatch "
            f"is fixed, not a transient event."
        )
        return incident

    if incident.source_stage.startswith("Stage 4"):
        if incident.blocking:
            # Blocking issues directly lose traffic -- weighted more
            # heavily than a non-blocking metadata corruption at the
            # same observed rate.
            if incident.peak_value > 0.10 or duration > 14:
                incident.priority = "P1"
            elif incident.peak_value > 0.03:
                incident.priority = "P2"
            else:
                incident.priority = "P3"
            incident.reasoning = (
                f"BLOCKING issue: peak {incident.peak_value:.1%} of traffic lost entirely, "
                f"sustained {duration} day(s)."
            )
        else:
            # Non-blocking: the ticket still gets classified, so this is
            # a metadata-integrity problem for downstream systems, not a
            # traffic-loss problem -- capped lower than blocking issues
            # at comparable rates, escalating only if it's large or very
            # persistent.
            if incident.peak_value > 0.25 or duration > 21:
                incident.priority = "P2"
            elif incident.peak_value > 0.10:
                incident.priority = "P3"
            else:
                incident.priority = "P4"
            incident.reasoning = (
                f"NON-BLOCKING issue: peak {incident.peak_value:.1%} of records affected "
                f"with a corrupted metadata field, sustained {duration} day(s) -- tickets "
                f"still get classified normally."
            )
        return incident

    # Stage 2 with CONFIRMED impact (accuracy actually dropped)
    if duration > 14 or incident.peak_value > 0.4:
        incident.priority = "P1"
    else:
        incident.priority = "P2"
    incident.reasoning = (
        f"Confirmed accuracy impact during a distribution shift lasting "
        f"{duration} day(s), peak PSI {incident.peak_value:.2f}."
    )
    return incident


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-csv", default=str(OBS_DIR / "drift_results_drift.csv"))
    parser.add_argument("--skew-summary", default=str(OBS_DIR / "skew_results" / "summary.json"))
    parser.add_argument("--dq-csv", default=str(OBS_DIR / "data_quality_results.csv"))
    args = parser.parse_args()

    incidents = []
    incidents += analyze_drift(Path(args.drift_csv))
    incidents += analyze_skew(Path(args.skew_summary))
    incidents += analyze_data_quality(Path(args.dq_csv))
    incidents = [score_priority(i) for i in incidents]

    priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
    incidents.sort(key=lambda i: priority_order.get(i.priority, 5))

    print(f"=== Alert digest: {len(incidents)} correlated incident(s) from "
          f"{sum(1 for i in incidents)} raw signal(s) ===\n")
    for i in incidents:
        print(f"[{i.priority}] {i.source_stage} -- {i.alert_type}")
        print(f"  Window: day {i.start_day}-{i.end_day}  |  Peak value: {i.peak_value:.3f}")
        print(f"  Impact: {i.impact_description}")
        print(f"  Priority reasoning: {i.reasoning}")
        print()

    n_confirmed = sum(1 for i in incidents if i.business_impact_confirmed)
    n_p1_p2 = sum(1 for i in incidents if i.priority in ("P1", "P2"))
    print(f"Summary: {len(incidents)} correlated incidents, {n_confirmed} with confirmed "
          f"business impact, {n_p1_p2} rated P1/P2 (page-worthy).")
    print("Everything else is a real, logged signal -- just not one that should "
          "interrupt someone at 2am.")


if __name__ == "__main__":
    main()
