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

## Results

*(Fill in after running `train_lora.py` and `evaluate.py` on a rented GPU
— the harness in `evaluate.py` writes `results/summary.json` with every
number below.)*

| Metric | Base (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| Overall accuracy | TBD | TBD |
| Invalid/unparseable output rate | TBD | TBD |
| Self-consistency (5 runs, T=0.7) | TBD | TBD |
| p50 latency | TBD | TBD |
| Avg output tokens | TBD | TBD |

**Per-category accuracy:** TBD (table from `results/summary.json`)

**Failure mode analysis:** TBD — pull the worst confusions from
`results/ft_results.json` (`correct: false`) and characterize them.
Expect the real signal to be in *which* categories get confused (e.g.
Network vs. Software when a ticket describes an app failing to connect)
rather than random noise — that distinction is worth a paragraph in
interviews.

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

## Repo structure

```
it-ticket-lora/
├── dataset/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── scripts/
│   ├── generate_dataset.py
│   ├── train_lora.py
│   └── evaluate.py
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