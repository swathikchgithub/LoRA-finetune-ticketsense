"""
model_utils.py

Shared utility for loading the fine-tuned ticket classifier and getting
predictions WITH a real confidence score -- not just the predicted label.

Why confidence matters for observability specifically (as opposed to the
original evaluate.py, which only needed accuracy): drift and skew often
show up as a quiet drop in confidence before accuracy visibly craters.
A model that's 95% accurate but suddenly averaging 60% confidence on its
correct answers is telling you something is off, even before enough
wrong predictions accumulate to move the accuracy number. This is a real,
production-relevant signal that pure offline accuracy testing doesn't
surface -- which is exactly the training-serving skew story for Stage 3.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"

CATEGORIES = [
    "Hardware", "Software", "Network", "Access_Account",
    "Password_Reset", "Email", "Security_Phishing", "Printer",
]

SYSTEM_PROMPT = (
    "You are an IT support ticket classifier. Read the ticket and respond "
    "with exactly one category from this list: "
    + ", ".join(CATEGORIES) + ". Respond with only the category name."
)


class TicketClassifier:
    """Wraps the fine-tuned model for repeated inference calls -- load once,
    predict many times, which is how a real serving process would work
    (not reloading the model per-request)."""

    def __init__(self, rank: int = 8, normalize_fn=None, adapter_dir=None):
        """
        rank: which LoRA checkpoint to load by the standard rank<N> naming
              convention (default 8 -- our ablation found it's the most
              parameter-efficient with no accuracy cost, see README).
              Ignored if adapter_dir is given explicitly.
        adapter_dir: explicit path override. Needed for scripts like
              auto_train_search.py, where configs vary by rank AND
              learning rate AND epochs -- the rank-only naming convention
              isn't enough to address every checkpoint the search produces.
        normalize_fn: optional text preprocessing function applied to the
              input BEFORE tokenization. This is the hook Stage 3 (training-
              serving skew) will use to simulate a serving path that
              normalizes text differently than training did -- e.g.
              truncating length, lowercasing, or stripping punctuation
              differently. Defaults to identity (no change) so Stage 1/2
              behave exactly like evaluate.py's proven pipeline.
        """
        if adapter_dir is None:
            adapter_dir = CHECKPOINTS_ROOT / f"lora-rank{rank}" / "final"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        self.model.eval()

        # Pre-compute the token ID for each category's FIRST token, so we
        # can read the confidence directly off the model's probability
        # distribution at the position where it commits to an answer,
        # without needing a second generation call.
        self.category_first_token = {}
        for cat in CATEGORIES:
            ids = self.tokenizer.encode(cat, add_special_tokens=False)
            self.category_first_token[cat] = ids[0]

        self.normalize_fn = normalize_fn or (lambda x: x)

    def predict(self, ticket_text: str):
        """Returns (predicted_category, confidence, latency_ms).

        confidence = the model's own probability estimate for the category
        it actually picked, read directly from the softmax over the next-
        token logits at the answer position. This is a single forward
        pass, not a generation loop, so it's fast and reflects the model's
        true internal certainty rather than something we infer indirectly.
        """
        normalized_text = self.normalize_fn(ticket_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalized_text},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(
            self.model.device
        )

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]  # logits for the very next token
            probs = F.softmax(next_token_logits, dim=-1)

            # Compare probability mass across just our known category
            # tokens (not the full ~150k vocabulary) -- this mirrors how
            # the model was trained to choose among exactly these 8 options.
            cat_probs = {
                cat: probs[tok_id].item()
                for cat, tok_id in self.category_first_token.items()
            }
            predicted = max(cat_probs, key=cat_probs.get)
            confidence = cat_probs[predicted]
        elapsed_ms = (time.perf_counter() - start) * 1000

        return predicted, confidence, elapsed_ms
