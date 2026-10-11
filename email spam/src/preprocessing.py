"""
Text Preprocessing and Feature Engineering Module for Email Spam Classifier.
"""

import re
import string
from typing import List, Tuple, Dict, Any

# Standard robust English stopwords list for reliable standalone execution
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        nltk_stopwords = set(stopwords.words('english'))
        STOPWORDS = nltk_stopwords
    except Exception:
        STOPWORDS = DEFAULT_STOPWORDS
    stemmer = PorterStemmer()
except Exception:
    STOPWORDS = DEFAULT_STOPWORDS
    stemmer = None


def extract_meta_features(text: str) -> Dict[str, Any]:
    """
    Extract structural and statistical metadata features from the raw email text.
    
    Spam emails frequently exhibit distinctive structural characteristics:
    - Higher character and word lengths
    - Excessive uppercase characters (URGENT, FREE, WINNER)
    - Punctuation anomalies (multiple exclamation marks !!!)
    - Presence of currency symbols ($ £ €) and digit clusters
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = (uppercase_count / char_count) if char_count > 0 else 0.0
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    currency_count = sum(text.count(sym) for sym in ['$', '£', '€', '₹', '¥'])
    digit_count = sum(1 for c in text if c.isdigit())
    
    # URL presence
    has_url = 1 if re.search(r'https?://\S+|www\.\S+', text) else 0

    return {
        'char_count': char_count,
        'word_count': word_count,
        'uppercase_ratio': uppercase_ratio,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'currency_count': currency_count,
        'digit_count': digit_count,
        'has_url': has_url
    }


def clean_text(text: str, stem: bool = True) -> str:
    """
    Perform comprehensive text preprocessing for NLP classification:
    1. Lowercase conversion
    2. URL, email, and digit extraction/normalization
    3. Punctuation stripping
    4. Tokenization and stopword removal
    5. Morphological stemming via PorterStemmer
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Normalize URLs, emails, phone numbers, and currency
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' emailtoken ', text)
    text = re.sub(r'[$£€₹¥]\s*\d+', ' moneytoken ', text)
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', ' phonetoken ', text)
    text = re.sub(r'\b\d+\b', ' numbertoken ', text)

    # 3. Strip punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Tokenize and filter stopwords
    tokens = text.split()
    filtered_tokens: List[str] = []
    for token in tokens:
        token_clean = token.strip()
        if len(token_clean) > 1 and token_clean not in STOPWORDS:
            if stem and stemmer:
                try:
                    token_clean = stemmer.stem(token_clean)
                except Exception:
                    pass
            filtered_tokens.append(token_clean)

    return " ".join(filtered_tokens)


def get_spam_trigger_words() -> List[str]:
    """Returns curated high-confidence spam trigger indicators for explanation."""
    return [
        'free', 'win', 'winner', 'won', 'cash', 'prize', 'urgent', 'claim',
        'selected', 'congratulations', 'credit', 'guaranteed', 'risk-free',
        'action required', 'limited time', 'call now', 'moneytoken', 'urltoken',
        'reply', 'text', 'txt', 'stop', 'offer', 'exclusive', 'bonus', 'million',
        'reward', 'apply now', 'billing', 'suspended', 'verify', 'password',
        'account alert', 'deposit', 'unclaimed', 'investment', 'instant'
    ]
