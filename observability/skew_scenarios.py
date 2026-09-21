"""
skew_scenarios.py -- Stage 3: training-serving skew definitions

Defines text-preprocessing functions that represent realistic ways a
SERVING pipeline can compute inputs differently than the TRAINING
pipeline did -- the exact failure mode the LinkedIn JD names explicitly.

Why this is the "most important" stage to get right: an offline eval
(evaluate.py) tests the model against test.jsonl using the SAME
preprocessing the model was trained on. If the actual serving code path
applies different preprocessing before calling the model, offline eval
never sees that code path at all -- it PASSES cleanly, because it was
never testing the thing that's actually broken. This is silent by
construction, not because anyone was careless with the eval.

SKEW SCENARIO CHOSEN: text truncation.
A very real, very boring bug: a legacy ticket-intake form field has a
character limit (say, from an old UI constraint) that nobody remembered
still applied when the serving integration was built, while the
training data was exported from a system with no such limit. The model
was trained on full ticket text; production silently hands it truncated
text. Nothing crashes. Nothing throws an error. It just quietly gets
worse at exactly the tickets where the cut-off text removed the actual
problem description.
"""


def identity(text: str) -> str:
    """The TRAINING-time preprocessing: no transformation at all. This is
    what evaluate.py and train_lora.py both implicitly use -- passing the
    raw ticket text straight to the model."""
    return text


def serving_truncation_skew(text: str, max_chars: int = 60) -> str:
    """The SERVING-time preprocessing bug: truncates to the first
    max_chars characters. 60 was chosen deliberately -- long enough that
    short tickets pass through untouched (so the bug isn't obviously
    broken for everything), short enough that many of our templates'
    actual problem description gets cut off, especially ones with a
    noise prefix eating into the budget (e.g. "Not sure who to ask, but "
    is 25 characters before the real ticket text even starts).
    """
    return text[:max_chars]
