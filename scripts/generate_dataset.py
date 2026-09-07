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
        "My laptop fan is making a loud grinding noise and the machine feels very hot.",
        "The HDMI port on my dock stopped outputting to my external monitor.",
        "My laptop hinge is loose and the screen flops backward on its own.",
        "Headset jack on my laptop stopped recognizing the microphone.",
        "The trackpad on my laptop is unresponsive to clicks in the corners.",
        "USB ports on my desktop stopped working after a power outage.",
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
        "PowerPoint hangs and stops responding whenever I insert a video clip.",
        "The reporting tool exports blank PDFs instead of the actual report.",
        "My browser extension for {app} stopped syncing since the last update.",
        "Getting a version mismatch error when opening a shared project file.",
        "The internal timesheet app logs me out every {n} minutes.",
        "{app} keeps asking me to re-authenticate every time I open a new tab.",
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
        "DNS seems to be failing, internal site names won't resolve for me.",
        "My laptop connects to wifi but shows no internet access.",
        "Video calls keep dropping every {n} minutes, seems network related.",
        "The office network is down building-wide since {n} minutes ago.",
        "VPN client crashes immediately on launch after the latest update.",
        "Can't reach the {app} server, ping times out from my machine.",
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
        "Requesting elevated permissions on {app} for the audit next week.",
        "My manager left the company and I need a new approver for access requests.",
        "Access request for the {app} sandbox environment has been pending for {n} days.",
        "I need to be added to the on-call rotation group in {app}.",
        "My contractor account expired but I still need access for {n} more days.",
        "Requesting VPN group access for the new remote team members.",
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
        "Reset link in the email is broken, clicking it gives a 404 error.",
        "I changed my password {n} days ago and now the new one isn't accepted either.",
        "Locked out of {app} specifically, other accounts still work fine.",
        "Need my account unlocked, I think I mistyped my password too many times.",
        "Self-service portal says my account doesn't exist when I try to reset.",
        "My password manager has the wrong password saved, need a manual reset.",
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
        "Emails I send are arriving with attachments stripped out.",
        "My inbox search isn't returning results from more than {n} days ago.",
        "Getting duplicate copies of every email that comes in.",
        "Out of office replies aren't triggering even though it's enabled.",
        "My email signature reverted to the default template after the update.",
        "Can't add a second calendar from {app} to my mailbox.",
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
        "A coworker forwarded me an email that looks like a credential harvesting attempt.",
        "I entered my password on a site that now looks suspicious in hindsight.",
        "Getting fake invoice emails from a domain that mimics our vendor's name.",
        "My account sent emails I never wrote, think it might be compromised.",
        "Received a QR code in an email asking me to scan and log in, seems off.",
        "IT never called me but I got a voicemail claiming to be from IT support.",
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
        "The printer is only printing half the page and cutting off content.",
        "Duplex printing stopped working, everything prints single-sided now.",
        "Print queue shows {n} stuck jobs and none of them will clear.",
        "Stapler attachment on the office printer jammed mid-job.",
        "Printer driver update broke printing from {app} specifically.",
        "Can't print from my laptop anymore, printer isn't showing in the list.",
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
    """Generates examples AND tracks which template produced each one, so
    we can later split by template rather than by individual example --
    this is what prevents train/test leakage (see split_by_template below).
    """
    rows = []
    for category, templates in TEMPLATES.items():
        seen = set()
        attempts = 0
        while len([r for r in rows if r["label"] == category]) < n_per_category and attempts < n_per_category * 20:
            attempts += 1
            template_idx = random.randrange(len(templates))
            t = templates[template_idx]
            text = render(t)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            # template_id uniquely identifies which of the 8 base sentence
            # patterns this example came from -- e.g. "Network_3" means
            # the 4th template in the Network category (0-indexed).
            rows.append({"text": text, "label": category, "template_id": f"{category}_{template_idx}"})
    random.shuffle(rows)
    return rows


def split_by_template(rows, train=0.7, val=0.15):
    """Splits by TEMPLATE, not by individual example. This is the fix for
    train/test leakage: with only 8 templates per category producing ~90
    examples each (via random.choice reuse), a plain random split of
    individual examples lets near-duplicate siblings of the same template
    land in both train and test -- inflating eval accuracy without the
    model actually generalizing to new phrasing.

    Instead: group examples by (category, template_id), shuffle the GROUPS,
    then assign whole groups to train/val/test. Every example generated
    from a given template ends up on exactly one side of the split.
    """
    from collections import defaultdict
    by_template = defaultdict(list)
    for r in rows:
        by_template[r["template_id"]].append(r)

    template_ids = list(by_template.keys())
    random.shuffle(template_ids)

    n = len(template_ids)
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val))

    train_templates = set(template_ids[:n_train])
    val_templates = set(template_ids[n_train:n_train + n_val])
    test_templates = set(template_ids[n_train + n_val:])

    train_rows = [r for tid in train_templates for r in by_template[tid]]
    val_rows = [r for tid in val_templates for r in by_template[tid]]
    test_rows = [r for tid in test_templates for r in by_template[tid]]

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            # Drop template_id before writing -- it's bookkeeping for the
            # split, not part of the actual training signal.
            f.write(json.dumps({"text": r["text"], "label": r["label"]}) + "\n")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "dataset"
    out_dir.mkdir(exist_ok=True)

    rows = generate(n_per_category=90)  # 90 * 8 categories = 720 examples
    train, val, test = split_by_template(rows)

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    print(f"Total: {len(rows)} | train: {len(train)} | val: {len(val)} | test: {len(test)}")
    from collections import Counter
    print("Category distribution (train):", Counter(r["label"] for r in train))