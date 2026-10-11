"""
Data Ingestion and Management Module.
Downloads or loads the SMS / Email Spam Collection dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_CSV_PATH = os.path.join(DATA_DIR, "spam.csv")
UCI_RAW_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")

# Established reliable mirrors for the UCI SMS Spam Collection
DATASET_URLS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
]


def ensure_dataset_exists() -> str:
    """
    Ensure the spam dataset exists locally in the data directory.
    If not present, downloads from authenticated or public mirrors.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_CSV_PATH):
        return DATA_CSV_PATH

    if os.path.exists(UCI_RAW_PATH):
        # Convert tab-separated SMSSpamCollection to standardized spam.csv
        df = pd.read_csv(UCI_RAW_PATH, sep='\t', names=['label', 'message'], encoding='latin-1')
        df.to_csv(DATA_CSV_PATH, index=False)
        return DATA_CSV_PATH

    # Download from reliable GitHub mirror first
    print("[*] Downloading SMS Spam Collection dataset...")
    downloaded = False
    for url in DATASET_URLS:
        try:
            print(f"[*] Trying mirror: {url}")
            temp_path = os.path.join(DATA_DIR, "downloaded_raw")
            urllib.request.urlretrieve(url, temp_path)
            
            if url.endswith(".zip"):
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                if os.path.exists(UCI_RAW_PATH):
                    df = pd.read_csv(UCI_RAW_PATH, sep='\t', names=['label', 'message'], encoding='latin-1')
                    df.to_csv(DATA_CSV_PATH, index=False)
                    downloaded = True
                    break
            else:
                # Tab-separated file
                df = pd.read_csv(temp_path, sep='\t', names=['label', 'message'], encoding='latin-1')
                # Clean up if header was read as data
                if df.iloc[0]['label'] in ['label', 'v1', 'Category']:
                    df = df.iloc[1:]
                df.to_csv(DATA_CSV_PATH, index=False)
                downloaded = True
                break
        except Exception as e:
            print(f"[!] Warning: Failed downloading from {url}: {e}")
            continue

    if not downloaded:
        print("[!] Generating initial curated reference dataset...")
        # Create a representative fallback dataset if network unavailable
        df = _generate_curated_fallback_dataset()
        df.to_csv(DATA_CSV_PATH, index=False)

    return DATA_CSV_PATH


def load_dataset() -> pd.DataFrame:
    """
    Load, validate, and standardize the spam classification dataset.
    Returns:
        pd.DataFrame with columns ['label', 'message', 'target']
        where target is 1 for 'spam' and 0 for 'ham'.
    """
    csv_path = ensure_dataset_exists()
    df = pd.read_csv(csv_path, encoding='latin-1')

    # Handle standard Kaggle (v1, v2) or UCI (label, message) schemas
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    elif 'Category' in df.columns and 'Message' in df.columns:
        df = df[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'message'})
    elif 'label' in df.columns and 'text' in df.columns:
        df = df.rename(columns={'text': 'message'})

    # Validate non-null values
    df = df.dropna(subset=['label', 'message'])
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['message'] = df['message'].astype(str).str.strip()

    # Filter to only valid spam/ham classes
    df = df[df['label'].isin(['ham', 'spam'])].copy()

    # Target: 1 = spam, 0 = ham
    df['target'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

    # Remove exact duplicate messages to avoid test leakage
    initial_len = len(df)
    df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
    print(f"[*] Loaded {len(df)} unique records (deduplicated from {initial_len}).")
    print(f"    - Ham (Legitimate): {sum(df['target'] == 0)} ({sum(df['target'] == 0)/len(df):.1%})")
    print(f"    - Spam: {sum(df['target'] == 1)} ({sum(df['target'] == 1)/len(df):.1%})")

    return df


def _generate_curated_fallback_dataset() -> pd.DataFrame:
    """Fallback sample messages for offline verification."""
    data = [
        ("ham", "Hey, are we still meeting for lunch today at 12:30? Let me know."),
        ("ham", "Please find attached the quarterly financial report for review."),
        ("ham", "Can you send me the slides from today's engineering presentation?"),
        ("ham", "Don't forget to submit your timesheets by 5 PM this Friday."),
        ("ham", "Great work on the release yesterday! Team dinner on Thursday."),
        ("ham", "I will be working from home tomorrow due to doctor appointment."),
        ("ham", "Are you available for a quick sync regarding the architecture?"),
        ("ham", "Thanks for your feedback on the pull request, I pushed changes."),
        ("ham", "Reminder: Dental cleaning appointment scheduled for Monday morning."),
        ("ham", "Hi mom, I will arrive home around 7 PM tonight. Love you."),
        ("spam", "URGENT! You have won a £1,000 cash prize or a guaranteed iPhone. Call 09061701461 to claim now."),
        ("spam", "Congratulations! Selected for exclusive $500 Walmart gift card. Click http://bit.ly/claim500 to redeem."),
        ("spam", "FREE entry in 2 a weekly comp to win £1000 cash! Text WIN to 80082. T&C apply."),
        ("spam", "Your bank account has been locked. Verify your credentials immediately at http://secure-login-update.com"),
        ("spam", "Loan approval guaranteed! No credit check required. Get up to $10,000 deposited in 24 hours. Call now!"),
        ("spam", "Hot singles in your area want to meet you! Click here to chat for free."),
        ("spam", "Claim your bonus! 500 free spins waiting in your casino wallet. Deposit $10 get $100."),
        ("spam", "FINAL NOTICE: IRS tax refund of $1,420 pending. Update payment details to receive transfer."),
        ("spam", "Double your bitcoin investment in 48 hours! Guaranteed passive income. Join our Telegram."),
        ("spam", "You have 1 new unread voicemail from mystery admirer. Dial *98 to listen now for $3.99/min.")
    ]
    return pd.DataFrame(data, columns=['label', 'message'])
