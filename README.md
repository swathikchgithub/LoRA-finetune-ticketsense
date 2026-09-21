# LoRA-finetune-ticketsense

**A rigorous base-vs-fine-tuned evaluation of LoRA on IT ticket classification.**

A small, end-to-end LoRA fine-tuning project: synthetic dataset generation,
training, and a rigorous base-vs-fine-tuned evaluation. Built to close a
specific gap — hands-on fine-tuning experience — as a complement to
production experience in LLM evaluation infrastructure and RAG/agentic
systems.

## What this is (and isn't)

This is a **portfolio / learning project**, not a claim of production
fine-tuning experience. The dataset is synthetic (~720 examples, 8
categories) because real ITSM ticket data is proprietary. The value of this
project isn't "I built a state-of-the-art classifier" — it's "I can walk
through every stage of a fine-tuning pipeline and, more importantly, I
know how to evaluate whether fine-tuning actually helped, which is where
most tutorials stop short."

## Task

Classify IT support tickets into one of 8 categories (Hardware, Software,
Network, Access_Account, Password_Reset, Email, Security_Phishing,
Printer) — chosen to mirror a real ITSM/ServiceNow taxonomy.

## Why this task

- Maps directly to ServiceNow/ITSM domain background, so the results are
  something I can sanity-check by eye, not just trust blindly.
- Classification-as-generation is a clean, well-scoped task for a first
  hands-on fine-tuning project: fast to iterate, easy to define ground
  truth, and the accuracy metric is unambiguous — which keeps the focus
  on the *fine-tuning mechanics and eval rigor*, not on wrestling with a
  fuzzy task definition.

## Model

**Qwen2.5-1.5B-Instruct** (Apache 2.0). Chosen because:
- Small enough to LoRA fine-tune on a single rented GPU (16-24GB) in
  under an hour, cheap enough to run this project for a few dollars.
- Instruction-tuned already, so the base model is a fair zero-shot
  baseline — the comparison isn't "fine-tuned vs. an unusable raw model."
- Actively used and well-documented, standard PEFT/HF tooling applies
  directly.

Swap-in alternative tried/considered: Llama-3.2-3B-Instruct, for a
larger-model comparison point (see `scripts/train_lora.py`, just change
`MODEL_NAME`).

## Method

1. **`scripts/generate_dataset.py`** — synthetic dataset generator.
   720 examples across 8 categories, with deliberately injected noise
   (typos, mixed formality, irrelevant filler) — clean templated data
   produces misleadingly high accuracy and hides the failure modes that
   are actually interesting to analyze. 70/15/15 train/val/test split.
2. **`scripts/train_lora.py`** — LoRA fine-tuning via HuggingFace
   `transformers` + `peft`. Classification framed as constrained
   generation (model outputs the category string) rather than a
   classification head, with the prompt tokens masked out of the loss.
   Every hyperparameter choice (rank, alpha, learning rate, target
   modules, epochs) is explained inline in the script — see that file
   for the reasoning, not just the values.
3. **`scripts/evaluate.py`** — the core of the project. Compares base
   (zero-shot prompted) vs. fine-tuned model on:
   - Overall and per-category accuracy, with a confusion breakdown
   - Self-consistency across repeated sampled runs (temperature > 0) —
     a genuinely fine-tuned model should agree with itself far more
     often than a prompted base model hedging between plausible labels
   - Latency (p50/p95) and output-token count as a cost proxy
   - Structured output validity — how often the base model fails to
     emit a clean parseable category at all
   - Saved per-example results for manual failure mode analysis

## A real bug I found and fixed: template-level data leakage

Worth documenting explicitly, because it's a more useful thing to be able
to discuss in an interview than a clean result would have been.

**What happened:** the first version of the dataset generator produced 90
examples per category from a pool of only 8 hand-written sentence
templates (filling in random numbers/app names and adding noise). The
train/val/test split was done by shuffling and cutting the 720 *finished
examples* — not by template. Since each template got reused ~11 times,
near-identical siblings of the same template (e.g. "WiFi keeps
disconnecting every 12 minutes" and "...every 31 minutes") ended up on
both sides of the split.

**The symptom:** after the first training run, `eval_loss` dropped to
~0.0001 — essentially zero — after only 3 epochs on ~500 examples. That's
a suspiciously strong result for so little training data, and it turned
out to be exactly that: not genuine generalization, but the model
recognizing test examples it had effectively already seen near-duplicates
of during training.

**The fix:** `generate_dataset.py` now tracks which template produced
each example and splits by *template group*, not by individual example —
every example from a given template lands entirely in train, val, or
test, never split across them. Verified zero template overlap between
train and test after the fix. Template variety per category was also
roughly doubled (8 → 14) to make the split more meaningful.

**Why this is worth including rather than hiding:** catching this kind of
leakage before trusting a number is exactly the evaluation instinct that
transfers from production ML systems work — a near-perfect metric should
raise suspicion before it's celebrated, not after.



## Results

| Metric | Base (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| Overall accuracy | 78.3% (90/115) | **91.3%** (105/115) |
| Invalid/unparseable output rate | 0.0% | 0.0% |
| Self-consistency (5 runs, T=0.7, 30 examples) | 83.3% | **96.7%** |
| p50 latency | 49ms | 100ms |
| p95 latency | 80ms | 194ms |
| Avg output tokens | 2.3 | 2.3 |

**Per-category accuracy:**

| Category | Base | Fine-tuned |
|---|---|---|
| Hardware | 100.0% (13/13) | 100.0% (13/13) |
| Software | 63.0% (17/27) | 74.1% (20/27) |
| Network | 100.0% (22/22) | 100.0% (22/22) |
| Access_Account | 75.0% (3/4) | 25.0% (1/4) |
| Password_Reset | 54.5% (6/11) | **100.0%** (11/11) |
| Email | 66.7% (10/15) | **100.0%** (15/15) |
| Security_Phishing | 50.0% (3/6) | **100.0%** (6/6) |
| Printer | 94.1% (16/17) | 100.0% (17/17) |

Note on `Access_Account`: only 4 test examples exist for this category (a
consequence of the template-level split putting most Access_Account
templates in train/val). One flipped prediction swings the percentage by
25 points — this number is too noisy to draw a real conclusion from, and
is flagged here rather than glossed over. A larger, more evenly
distributed test set would fix this.

**On latency:** the fine-tuned model is ~2x slower per prediction (100ms
vs 49ms p50). This isn't a fine-tuning downside in general — it's because
the LoRA adapter is still being applied as a separate set of matrices on
top of the frozen base weights at inference time here. In a real
deployment you'd call `model.merge_and_unload()` (PEFT's built-in method)
after training, which folds the adapter weights directly into the base
model's weights — at that point inference speed matches the base model
exactly, with zero LoRA overhead. Worth stating explicitly: **LoRA has an
inference cost until merged**, and this project's own numbers demonstrate
it directly.

## LoRA rank ablation: 8 vs 16 vs 32

Ran the identical pipeline at three ranks to test whether more adapter
capacity actually helps on this task/dataset size, rather than picking
rank 16 by rule of thumb alone.

| Rank | Trainable % | Best eval_loss | Test accuracy | Fine-tuned consistency |
|---|---|---|---|---|
| 8 | 0.59% | 0.096 | 92.2% (106/115) | **100%** |
| 16 | 1.18% | **0.067** (best loss) | 91.3% (105/115) | 93.3% |
| 32 | 2.34% | 0.204 (worst loss) | **93.0%** (best accuracy, 107/115) | 93.3% |

**Finding — loss and accuracy don't agree on which rank is "best," and
that's a real, worth-understanding fact rather than a bug:** rank 16 had
the lowest loss but not the highest accuracy; rank 32 had the worst loss
but the highest accuracy. Loss measures how confident the model's full
probability distribution was at every position; accuracy only checks
whether the single most-likely token happened to be correct. A less
confident model can still pick the right answer slightly more often,
especially on a test set this small (115 examples), where a couple of
flipped predictions swing the percentage by 1-2 points either way.

**Two honest caveats, not glossed over:**
- These three runs were **not seeded** (`torch.manual_seed()` was added
  to `train_lora.py` after these ran), so part of the rank-to-rank
  difference is plausibly random-initialization noise rather than a pure
  rank effect. Rank 32's odd Network-category regression (72.7%, vs.
  100% at rank 8 and 16, confused with Hardware) is a plausible symptom
  of this rather than a real "rank 32 hurts Network specifically" effect.
- A 2-point accuracy spread (91.3% to 93.0%) on 115 test examples is
  within noise, not a strong signal — a rigorous read of this ablation
  is "these three ranks perform roughly equivalently," not "rank 32 won."

**Practical conclusion:** given accuracy is statistically indistinguishable
across all three ranks here, **rank 8 is the better real-world choice** —
it uses roughly a quarter of rank 32's trainable parameters and about half
of rank 16's, for accuracy that's no worse (and, on this run, its
fine-tuned model was also the most self-consistent, at 100%). The
takeaway for future work: pick the smallest LoRA rank that gets the job
done, not the largest one that's affordable — bigger isn't free, and on
a small dataset it actively risks overfitting without a compensating
accuracy gain.

## Failure mode analysis

All 10 of the fine-tuned model's test-set errors trace back to two root
causes — both cases of genuine label ambiguity in the dataset rather than
random model confusion:

1. **6/10 errors**: tickets phrased like "*[app] keeps asking me to
   re-authenticate every time I open a new tab*" are labeled `Software`
   in the dataset, but the model consistently predicts `Password_Reset`.
   This is a defensible disagreement, not a mistake — a ticket about
   repeated re-authentication prompts genuinely sits on the boundary
   between "the app is buggy" and "something's wrong with my
   login/session," and a real support team might route it either way.
2. **3/10 errors**: "*Can't access the finance dashboard, getting a
   permission denied error*" is labeled `Access_Account`, predicted
   `Software` — same pattern, "permission denied" is ambiguous between
   an access problem and an app bug.

The takeaway: the model's errors are **concentrated on genuinely fuzzy
category boundaries**, not scattered randomly across unrelated
categories. That's a meaningfully different (and better) finding than
"the model got some wrong" — it suggests the remaining error rate has
more to do with label design than model capability, and the fix would be
tightening the category definitions or adding a tie-breaking rule to the
system prompt, not more training data.


## What I'd do differently at production scale

Worth saying explicitly, since this is a toy project:
- Real deployment needs human-labeled data, not synthetic — synthetic
  data teaches the pipeline, not the true label distribution or edge
  cases a real support queue has.
- 500-1000 examples is enough to demonstrate the LoRA mechanics; a
  production classifier would want several thousand labeled examples
  per category, active-learning loops on low-confidence predictions,
  and periodic retraining as ticket patterns shift.
- I'd add calibration analysis (is the model's confidence trustworthy
  enough to auto-route high-confidence tickets and escalate low-confidence
  ones to a human?) — that's the eval question that actually matters for
  a routing system in production, more than raw accuracy.

## Post-deployment observability layer

Everything above covers pre-deployment: training, evaluation, and
ablation. This section extends the project past that point, on purpose
-- the fine-tuning work answers "does the model work," but says nothing
about what happens to it after deployment. This layer simulates 90 days
of production traffic against the deployed model and builds the
monitoring that would catch problems offline evaluation cannot see:
data drift, training-serving skew, and data quality degradation.

### Stage 1: Traffic simulation + telemetry

`observability/simulate_traffic.py` generates a stream of realistic
tickets over a simulated 90-day period, runs each through the fine-tuned
model (rank-8 checkpoint), and logs every prediction -- input text, true
category, predicted category, a real confidence score (read directly off
the model's softmax, not inferred), and latency -- to a SQLite database
built for time-series queries.

Baseline run (no injected issues): mean daily accuracy 97.8%, std 3.1%
purely from sampling noise at 20 tickets/day. This noise band matters --
it's the threshold everything downstream is calibrated against, instead
of an arbitrary guess.

### Stage 2: Data drift detection (PSI / KL divergence)

`observability/drift_detection.py` computes Population Stability Index
and KL divergence between the training-time category distribution and a
rolling 7-day window of the model's *predictions* -- deliberately using
predictions, not ground truth, because that's the point: this signal
requires zero labels and is available in real time, unlike accuracy.

A second simulation run (`--scenario drift`) shifts the category mix
starting day 45, weighted toward `Access_Account` and `Software` --
chosen because the original held-out test set (only 4 and 27 examples
respectively) showed weak `Access_Account` accuracy (25%), flagged in
this README as too small a sample to trust.

**What actually happened is a more valuable finding than what was
designed for, and it reproduced on a second, independent run** (different
pod, same code, fresh random seed for traffic generation): PSI detected
the shift decisively and correctly both times -- rising from a baseline
of 0.01-0.14 to a sustained 0.4-0.65, crossing the 0.25 "significant
shift" threshold on day 50 in both runs. But **rolling accuracy never
dropped below the alert threshold in either run** -- it stayed in the
93-98% range throughout. At production scale (hundreds of real examples
per category, vs. 4 in the original test set), the model handles
`Access_Account` fine; the 25% figure was small-sample noise, not a real
weakness -- confirming, with real data instead of a caveat, the exact
concern already flagged in the ablation results above.

**Why this is the more useful result to report, not a failed
experiment:** it's a clean, real example of exactly the alert-fatigue
problem Stage 5 has to solve. A strong, correct, threshold-crossing PSI
alert (0.67, far past 0.25) corresponded to zero actual harm. Paging
on-call for every PSI breach here would mean ~40 consecutive days of
alerts for a shift that never degraded anything -- a concrete, measured
example of why distribution-shift alerts need business-impact
prioritization, not just a statistical threshold.

Separately, `drift_detection.py` also models realistic label delay via
`--label-delay-days` (default 5) -- ground truth from a human reviewer
arrives days later in reality, not instantly, and the script computes how
many days PSI would surface a shift before a label-delayed accuracy
monitor could. **Worth being precise about what this run did and didn't
show:** because accuracy never crossed its alert threshold in this
experiment, there's no real lead-time number to report here -- the
mechanism was validated separately against synthetic data with a
deliberately-injected accuracy drop (confirming a 4-day lead time in that
controlled test), but that number describes the synthetic validation, not
this model's actual production behavior. The honest finding from the
real run is the one above: PSI fired decisively, accuracy never needed
to.

### Coming next: Stage 3 (training-serving skew), Stage 4 (data quality
monitoring), Stage 5 (dashboard + alerting with explicit alert-fatigue
prioritization).

## Repo structure

```
LoRA-finetune-ticketsense/
├── dataset/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── scripts/
│   ├── generate_dataset.py
│   ├── train_lora.py
│   └── evaluate.py
├── observability/
│   ├── model_utils.py        # shared model loading + confidence scoring
│   ├── simulate_traffic.py   # Stage 1: 90-day traffic simulation
│   ├── drift_detection.py    # Stage 2: PSI/KL drift detection
│   └── telemetry.db          # gitignored, produced by simulate_traffic.py
├── results/            # produced by evaluate.py
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py     # already run, dataset/ is populated
python scripts/train_lora.py           # needs a GPU — RunPod/Lambda/Colab
python scripts/evaluate.py             # base vs. fine-tuned comparison
```
