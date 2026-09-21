"""
data_quality.py -- Stage 4: ingestion-time data quality checks

WHY THIS IS A SEPARATE LAYER FROM STAGES 1-3: those stages watch what the
MODEL does with data it receives. This stage watches whether the DATA
ITSELF is trustworthy BEFORE it reaches the model at all. That
distinction matters operationally -- if something goes wrong, "the data
pipeline is broken upstream" and "the model is behaving badly" call for
completely different responses and completely different owners. A good
observability system has to be able to tell those two apart, not lump
every anomaly into one undifferentiated alert stream.

Three realistic data quality failure modes, matching what actually shows
up in production ML systems:
  1. NULL/missing required data -- a field that must be present is empty.
     BLOCKING: if the ticket text itself is null/empty, there is nothing
     for the model to classify -- this ticket should be rejected at
     ingestion, never even sent to the model.
  2. SILENT DEFAULT -- an upstream system stops sending real values for a
     field and starts sending a placeholder/sentinel instead, without
     erroring. This is the most dangerous category precisely because
     nothing crashes -- the pipeline looks healthy while quietly losing
     information.
  3. SCHEMA VIOLATION -- a field's value falls outside its expected
     set/format, signaling the upstream contract changed (a new
     enum value appeared that ingestion code was never updated to expect).
"""

REQUIRED_TEXT_MIN_LENGTH = 5

KNOWN_PRIORITIES = {"P1", "P2", "P3", "P4", "P5"}
SILENT_DEFAULT_PRIORITY = "UNSET"  # what a broken upstream integration sends instead of a real priority

KNOWN_SOURCE_SYSTEMS = {"web_portal", "email_intake", "mobile_app", "phone_agent"}


def check_null_text(record: dict):
    """BLOCKING issue: no ticket text means nothing to classify. Returns
    an issue dict if the ticket should be rejected before reaching the
    model, else None."""
    text = record.get("ticket_text")
    if text is None or len(text.strip()) < REQUIRED_TEXT_MIN_LENGTH:
        return {"issue_type": "null_or_empty_text", "severity": "blocking",
                "field": "ticket_text", "value": repr(text)}
    return None


def check_silent_default_priority(record: dict):
    """NON-BLOCKING issue: the ticket can still be classified, but the
    priority field has silently reverted to a sentinel value instead of
    a real one -- a real signal something upstream broke, even though
    nothing here would crash or error."""
    priority = record.get("priority")
    if priority == SILENT_DEFAULT_PRIORITY:
        return {"issue_type": "silent_default_priority", "severity": "non_blocking",
                "field": "priority", "value": priority}
    return None


def check_schema_violation_source_system(record: dict):
    """NON-BLOCKING issue: source_system holds a value outside the known
    enum -- the upstream schema/contract changed without this ingestion
    code being updated to expect it."""
    source = record.get("source_system")
    if source is not None and source not in KNOWN_SOURCE_SYSTEMS:
        return {"issue_type": "schema_violation_source_system", "severity": "non_blocking",
                "field": "source_system", "value": source}
    return None


def validate_ticket(record: dict):
    """Runs all checks and returns the list of issues found (empty list
    if clean). Callers should check whether any issue has
    severity == 'blocking' and, if so, skip sending the record to the
    model entirely -- that's the core behavior this stage demonstrates:
    catching bad data BEFORE it wastes a model call or produces a
    meaningless prediction.
    """
    checks = [check_null_text, check_silent_default_priority, check_schema_violation_source_system]
    issues = []
    for check in checks:
        result = check(record)
        if result is not None:
            issues.append(result)
    return issues


def has_blocking_issue(issues: list) -> bool:
    return any(i["severity"] == "blocking" for i in issues)
