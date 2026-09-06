"""
generate_dataset.py

Generates a synthetic IT support ticket classification dataset.

WHY SYNTHETIC DATA (be ready to defend this in interviews):
- Real ITSM ticket data is proprietary / contains PII, so it can't ship in a
  public repo. Synthetic data is the honest, defensible choice for a
  portfolio project — the point is to demonstrate the *pipeline*
  (data -> training -> eval), not to claim a state-of-the-art classifier.
- We deliberately inject noise (typos, mixed formality, incomplete tickets,
  ambiguous category boundaries) because clean, templated synthetic data
  produces misleadingly high accuracy and teaches you nothing about failure
  modes. A README section calls this out explicitly.

CATEGORIES (chosen to map to real ITSM/ServiceNow taxonomies):
  1. Hardware
  2. Software
  3. Network
  4. Access_Account
  5. Password_Reset
  6. Email
  7. Security_Phishing
  8. Printer

Each ticket = {"text": <ticket body>, "label": <category>}
"""

import json
import random
from pathlib import Path

random.seed(42)  # reproducibility — call this out in the README

CATEGORIES = [
    "Hardware",
    "Software",
    "Network",
    "Access_Account",
    "Password_Reset",
    "Email",
    "Security_Phishing",
    "Printer",
]

# Each category gets: subjects (short phrases), body templates, and a pool of
# realistic noise fragments (typos, urgency markers, irrelevant context) that
# get randomly mixed in. This is what keeps the task non-trivial.

TEMPLATES = {
    "Hardware": [
        "My laptop won't turn on, I've tried holding the power button for {n} seconds and nothing happens.",
        "The monitor on my desk is flickering constantly since this morning.",
        "My keyboard is missing several keys and I need a replacement ASAP.",
        "Laptop battery drains from 100% to 0% in under an hour, seems defective.",
        "The docking station in conference room {n} isn't charging any laptops.",
        "My mouse cursor jumps around randomly, might be a hardware issue with the trackpad.",
        "Requesting a new laptop, mine has a cracked screen from a drop.",
        "The webcam on my work laptop stopped working after the last update, might be hardware.",
    ],
    "Software": [
        "Excel keeps crashing whenever I try to open a file larger than {n}MB.",
        "I can't install the new version of {app}, it fails at {n}% every time.",
        "The CRM application freezes when I click on the reports tab.",
        "My IDE won't launch, throws an error about a missing dependency.",
        "{app} is showing a blank white screen instead of loading.",
        "Getting a licensing error when I try to open {app}, says my seat expired.",
        "The internal dashboard app throws a 500 error on the analytics page.",
        "Need {app} installed on my machine for the project starting {n} days from now.",
    ],
    "Network": [
        "WiFi keeps disconnecting every {n} minutes on the 3rd floor.",
        "I can't connect to the VPN, it just times out after trying for a while.",
        "Internet is extremely slow in the building today, pages take forever to load.",
        "Ethernet port in my office doesn't seem to be getting any signal.",
        "The VPN connects but I can't reach any internal tools once connected.",
        "Guest wifi network isn't showing up in the list of available networks.",
        "Getting intermittent packet loss on calls, video keeps freezing.",
        "Can't access the shared drive from home even though VPN shows connected.",
    ],
    "Access_Account": [
        "I need access to the {app} project folder, my manager approved it {n} days ago.",
        "Can't access the finance dashboard, getting a permission denied error.",
        "Requesting admin rights on my laptop to install dev tools.",
        "My account got locked out after I changed teams, need access restored.",
        "New hire starting {n} days from now needs standard access provisioned.",
        "I was removed from the {app} workspace by mistake, please re-add me.",
        "Need read access to the shared repository for the {app} project.",
        "My badge access to the {n}th floor server room was revoked incorrectly.",
    ],
    "Password_Reset": [
        "I forgot my password and the self-service reset link isn't working.",
        "Locked out of my account after too many failed login attempts, need a reset.",
        "Password reset email never arrived, checked spam folder too.",
        "Need to reset my password, it expired and the portal won't let me set a new one.",
        "MFA app got reinstalled on a new phone, can't log in anymore, need reset.",
        "My temporary password isn't working, requesting a new one be issued.",
        "Can't reset password, security questions aren't matching what I remember setting.",
        "Password expired while I was on leave, portal locked me out entirely.",
    ],
    "Email": [
        "Not receiving any emails since this morning, outbox is also stuck.",
        "My calendar invites aren't syncing with {app}, meetings are missing.",
        "Emails from {app} are going straight to spam, need that whitelisted.",
        "Can't send attachments over {n}MB, keeps bouncing back.",
        "My shared mailbox for the support team isn't showing new tickets.",
        "Autoreply won't turn off even though I disabled it in settings.",
        "Getting a mailbox full error but I'm nowhere near the storage limit.",
        "Distribution list {app}-team isn't delivering to all members.",
    ],
    "Security_Phishing": [
        "Got a suspicious email asking me to verify my password, might be phishing.",
        "I think I clicked a bad link in an email, want to report it and check my account.",
        "Received a text claiming to be from IT asking for my MFA code, seems like smishing.",
        "Someone is asking for my login over a call claiming to be IT support, is this legit?",
        "Flagging an email that impersonates our CEO asking for an urgent wire transfer.",
        "My account showed a login from a location I've never been to, might be compromised.",
        "Downloaded an attachment before realizing the sender domain looked fake.",
        "Getting repeated phishing attempts on my work email this week, need it reported.",
    ],
    "Printer": [
        "Printer on the {n}th floor is jammed and won't clear even after I opened the tray.",
        "Can't print in color anymore, only black and white comes out.",
        "The printer shows offline even though it's turned on and connected.",
        "Print job has been stuck in the queue for {n} minutes.",
        "Scanner attached to the printer isn't sending scans to my email.",
        "Toner light is blinking red but we just replaced the cartridge.",
        "My print jobs are coming out with streaks across every page.",
        "Wireless printing stopped working after the network change last week.",
    ],
}

APPS = ["Salesforce", "Slack", "Zoom", "Tableau", "Jira", "Okta", "Workday", "SAP", "ServiceNow", "GitHub"]

NOISE_PREFIXES = [
    "", "", "", "Hi team, ", "Hey, ", "Urgent - ", "Following up again - ",
    "Not sure who to ask, but ", "Quick question, ",
]

NOISE_SUFFIXES = [
    "", "", "", " Please advise.", " This is blocking my work.",
    " Thanks in advance.", " Let me know what you need from me.",
    " This has been going on for a couple days now.",
]

TYPO_SWAPS = [("the", "teh"), ("issue", "isue"), ("connect", "conect"), ("password", "passowrd")]


def maybe_typo(text: str) -> str:
    if random.random() < 0.12:
        for a, b in TYPO_SWAPS:
            if a in text and random.random() < 0.5:
                text = text.replace(a, b, 1)
                break
    return text


def render(template: str) -> str:
    text = template.format(
        n=random.randint(2, 45),
        app=random.choice(APPS),
    )
    text = random.choice(NOISE_PREFIXES) + text + random.choice(NOISE_SUFFIXES)
    return maybe_typo(text)


def generate(n_per_category: int = 90):
    rows = []
    for category, templates in TEMPLATES.items():
        seen = set()
        attempts = 0
        while len([r for r in rows if r["label"] == category]) < n_per_category and attempts < n_per_category * 20:
            attempts += 1
            t = random.choice(templates)
            text = render(t)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": category})
    random.shuffle(rows)
    return rows


def split(rows, train=0.7, val=0.15):
    n = len(rows)
    n_train = int(n * train)
    n_val = int(n * val)
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "dataset"
    out_dir.mkdir(exist_ok=True)

    rows = generate(n_per_category=90)  # 90 * 8 categories = 720 examples
    train, val, test = split(rows)

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    print(f"Total: {len(rows)} | train: {len(train)} | val: {len(val)} | test: {len(test)}")
    from collections import Counter
    print("Category distribution (train):", Counter(r["label"] for r in train))
