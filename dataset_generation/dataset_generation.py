# Creating the synthetic dataset of 800 payment disputes as requested
import random, csv, os, re, math
from datetime import datetime
import pandas as pd

random.seed(42)

# Distribution
n = 800
dist = {
    "DUPLICATE_CHARGE": int(0.25 * n),   # 200
    "FAILED_TRANSACTION": int(0.30 * n), # 240
    "FRAUD": int(0.15 * n),              # 120
    "REFUND_PENDING": int(0.20 * n),     # 160
    "OTHERS": int(0.10 * n)              # 80
}
# Adjust any rounding
total_assigned = sum(dist.values())
if total_assigned != n:
    dist["OTHERS"] += (n - total_assigned)

# Helper utilities for generating varied descriptions
merchants = ["grocery store", "petrol pump", "restaurant", "online marketplace", "mobile recharge", "utility bill", "electronics shop", "pharmacy", "taxi", "clothing store", "ticketing site", "subscription service"]
amounts = ["₹1200", "₹499", "₹2500", "₹59.99", "₹2,300.00", "₹750", "₹3,500", "₹99", "₹450", "₹1,999"]
times = ["yesterday", "today", "this morning", "last night", "two days ago", "on 10/09/2025", "a week ago", "just now"]

def inject_typos(s, prob=0.12):
    s = list(s)
    for i in range(len(s)):
        if random.random() < prob:
            op = random.choice(["drop","dup","swap"])
            if op=="drop":
                s[i] = ""  # drop char
            elif op=="dup":
                s[i] = s[i]*2
            elif op=="swap" and i < len(s)-1:
                s[i], s[i+1] = s[i+1], s[i]
    return "".join(s).replace("  "," ").strip()

def choose(template_list):
    return random.choice(template_list)

# Templates per category
templates = {
    "DUPLICATE_CHARGE": [
        "I got charged twice for the same {merchant} of {amount} {time}.",
        "Double charge on card for {amount} at {merchant}. Please refund.",
        "I was debited two times, same txn, same amount. charged twice!",
        "Two debits for one purchase at {merchant} minutes apart.",
        "duplicated charge for {amount} — I only did the payment once.",
        "Was charged again for the same order, looks like a duplicate payment.",
        "charged twice. 2 entries for same txn id at {merchant}.",
        "I see a duplicate transaction of {amount} for same merchant, pls help."
    ],
    "FAILED_TRANSACTION": [
        "Transaction shows FAILED but money was debited from my account.",
        "My payment failed on gateway but amount got deducted, pls reverse.",
        "NEFT failed but amount not reversed yet.",
        "UPI failed but I got debited. Status shows FAILED.",
        "Payment failed but bank debited me; transaction status is FAIL.",
        "Payment went through at my end but merchant shows failed—I've been charged.",
        "Failed transaction but money taken out of my savings account."
    ],
    "FRAUD": [
        "I did not make this payment. This is fraud.",
        "Unauthorized transaction on my card — suspicious charge I didn't do.",
        "This looks like fraud, I didn't authorize this debit.",
        "Not my transaction. Someone used my card, it's unauthorised.",
        "Suspicious txn on my account, please investigate — not me.",
        "I didn't authorize this UPI transfer. This is definitely fraud."
    ],
    "REFUND_PENDING": [
        "Still waiting for refund after canceled transaction.",
        "Refund pending for a week after I cancelled the order.",
        "Refund not received, I've been waiting for days.",
        "Refund for canceled order not processed yet.",
        "Refund hasn't come back to my account after the seller cancelled.",
        "Refund waiting, merchant says refunded but not in my bank."
    ],
    "OTHERS": [
        "Payment stuck in pending, not sure if merchant got it.",
        "Chargeback requested as I don't recognize this card txn.",
        "My NEFT credited to wrong beneficiary per my statement.",
        "EMI card was charged twice this month (not sure why).",
        "Got two UPI debit messages for one QR scan — unsure what happened.",
        "Payment unclear — says processed but merchant not received."
    ]
}

# Desired verification probabilities
dup_verified_prob = 0.95   # for DUPLICATE_CHARGE
failed_verified_prob = 0.90 # for FAILED_TRANSACTION

rows = []
current_id = 1001

# Function to build description with some randomization and typos per instance
def build_description(cat):
    base = choose(templates[cat])
    # format placeholders if any
    base = base.format(merchant=choose(merchants), amount=choose(amounts), time=choose(times))
    # Randomly vary case, punctuation, length
    if random.random() < 0.25:
        base = base.lower()
    if random.random() < 0.15:
        base = base + " pls help"
    if random.random() < 0.12:
        base = inject_typos(base, prob=0.10)
    # Occasionally add more detail
    if random.random() < 0.18:
        extra = random.choice([
            "I have the SMS and bank statement.",
            "Transaction IDs: " + str(random.randint(100000,999999)),
            "I contacted merchant but no response.",
            "Please reverse asap, this is urgent.",
            "I have screenshots if needed."
        ])
        base = base + " " + extra
    return base

# Generate rows per category with flags set logically
for cat, count in dist.items():
    for i in range(count):
        desc = build_description(cat)
        # Initialize flags
        is_verified_duplicate = 0
        is_verified_failed = 0
        contains_fraud_keyword = 0
        contains_refund_keyword = 0
        contains_duplicate_keyword = 0
        
        # For DUPLICATE_CHARGE
        if cat == "DUPLICATE_CHARGE":
            # Mostly verified duplicate
            is_verified_duplicate = 1 if random.random() < dup_verified_prob else 0
            # include duplicate keyword in most descriptions
            if random.random() < 0.92:
                # ensure description contains a duplicate keyword
                if not re.search(r"\btwice\b|\bdouble\b|\bduplicate\b|\bcharged again\b|\bcharged twice\b", desc, re.I):
                    desc = "Double charge: " + desc
            # contains duplicate keyword computed from text later
        elif cat == "FAILED_TRANSACTION":
            is_verified_failed = 1 if random.random() < failed_verified_prob else 0
            # sometimes mention refund too
            if random.random() < 0.25:
                desc = desc + " refund not yet received."
        elif cat == "FRAUD":
            # ensure fraud keywords present
            if not re.search(r"\bfraud\b|\bunauthoriz|suspicious|didn't make this payment|not me\b", desc, re.I):
                desc = "Unauthorised transaction: " + desc
        elif cat == "REFUND_PENDING":
            # ensure refund phrasing
            if not re.search(r"\brefund\b|\bwaiting for money\b|\bcancel", desc, re.I):
                desc = "Refund pending: " + desc
        else:
            # OTHERS: keep ambiguous; small chance of keywords
            if random.random() < 0.08:
                desc = desc + " refund pending."
            if random.random() < 0.06:
                desc = "Suspicious txn: " + desc
        
        # Compute contains_* flags by searching text to ensure consistency
        txt = desc.lower()
        if re.search(r"\bfraud\b|\bunauthoriz|suspicious|didn't make this payment|not me\b|unauthorised", txt):
            contains_fraud_keyword = 1
        if re.search(r"\brefund\b|\bwaiting for money\b|\bcancelled\b|\bcanceled\b|\bnot received\b|\brefund pending\b", txt):
            contains_refund_keyword = 1
        if re.search(r"\btwice\b|\bdouble\b|\bduplicate\b|\bcharged again\b|\bcharged twice\b|\bduplicated\b", txt):
            contains_duplicate_keyword = 1
        
        # Keep logical consistency tweaks:
        # DUPLICATE_CHARGE should almost always have is_verified_duplicate=1 and contain duplicate keyword
        if cat == "DUPLICATE_CHARGE":
            if contains_duplicate_keyword == 0:
                # force keyword and description consistency
                desc = "Duplicate charge: " + desc
                contains_duplicate_keyword = 1
            # ensure most are verified (already set probabilistically)
        # FRAUD should have contains_fraud_keyword
        if cat == "FRAUD" and contains_fraud_keyword == 0:
            desc = "Fraud: " + desc
            contains_fraud_keyword = 1
        # FAILED_TRANSACTION should normally have is_verified_failed=1 and mention failed
        if cat == "FAILED_TRANSACTION":
            if is_verified_failed == 1 and "failed" not in txt:
                desc = "Failed transaction: " + desc
        # REFUND_PENDING should contain refund keyword
        if cat == "REFUND_PENDING" and contains_refund_keyword == 0:
            desc = "Refund pending: " + desc
            contains_refund_keyword = 1
        
        rows.append({
            "dispute_id": f"D{current_id}",
            "description": desc,
            "is_verified_duplicate": int(is_verified_duplicate),
            "is_verified_failed": int(is_verified_failed),
            "contains_fraud_keyword": int(contains_fraud_keyword),
            "contains_refund_keyword": int(contains_refund_keyword),
            "contains_duplicate_keyword": int(contains_duplicate_keyword),
            "true_category": cat
        })
        current_id += 1

# Shuffle rows to mix categories
random.shuffle(rows)

# Create DataFrame
df = pd.DataFrame(rows, columns=[
    "dispute_id", "description", "is_verified_duplicate", "is_verified_failed",
    "contains_fraud_keyword", "contains_refund_keyword", "contains_duplicate_keyword", "true_category"
])

# Quick sanity checks: counts per category and flag correlations
counts = df['true_category'].value_counts().to_dict()

# Save to CSV
out_path = "/mnt/data/disputes_800.csv"
df.to_csv(out_path, index=False)

# Show first 20 rows and distribution summary
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("First 200 disputes preview", df.head(200))

counts, out_path, df.shape[0]
