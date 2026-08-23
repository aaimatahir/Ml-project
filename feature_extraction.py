"""
Person 1 module: URL feature extraction.

Computes the same 24 features used to build feature_dataset_v2.csv, directly
from a raw URL string, with no network access. Used both to (re)build the
training dataset from the 'url' column and to score a URL submitted live
through the Flask app, so training and inference stay consistent.

Formulas for url_length, dot/hyphen/slash/question/equals/ampersand/percent/
at/colon/double-slash counts, digit/letter counts and ratios, domain/path/
query length, subdomain_count, has_https, has_ip and url_entropy were
reverse-engineered from feature_dataset_v2.csv and verified to match exactly.
special_char_count and suspicious_keyword_count could not be recovered from
the data (no single consistent formula fit), so they use clear definitions
of our own -- the dataset is regenerated from these same definitions so
training and serving stay consistent.
"""

import math
import re
from collections import Counter

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "bank", "password", "webscr", "ebayisapi", "paypal",
    "suspend", "urgent", "click", "invoice", "billing", "security",
    "recover", "unlock", "limited", "alert", "authenticate", "wallet",
]

SPECIAL_CHARS = set("!\"#$^*()_+~`{}[]|\\;'<>,")

IP_RE = re.compile(r"(\d{1,3}\.){3}\d{1,3}")


def _split_domain_path_query(url: str):
    if url.startswith("https://"):
        rest = url[len("https://"):]
    elif url.startswith("http://"):
        rest = url[len("http://"):]
    else:
        rest = url

    domain = rest.split("/", 1)[0]
    remainder = rest[len(domain):]

    # drop the fragment entirely before splitting path/query
    remainder = remainder.split("#", 1)[0]

    if "?" in remainder:
        path, query = remainder.split("?", 1)
    else:
        path, query = remainder, ""

    return domain, path, query


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_features(url: str) -> dict:
    url = url.strip()
    length = len(url)

    domain, path, query = _split_domain_path_query(url)

    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    special_char_count = sum(1 for c in url if c in SPECIAL_CHARS)

    labels = [p for p in domain.split(".") if p]
    subdomain_count = max(0, len(labels) - 2)

    lowered = url.lower()
    suspicious_keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lowered)

    features = {
        "url_length": length,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "special_char_count": special_char_count,
        "has_https": 1 if url.startswith("https://") else 0,
        "subdomain_count": subdomain_count,
        "has_ip": 1 if IP_RE.search(url) else 0,
        "suspicious_keyword_count": suspicious_keyword_count,
        "digit_count": digit_count,
        "letter_count": letter_count,
        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equals_count": url.count("="),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "at_symbol_count": url.count("@"),
        "colon_count": url.count(":"),
        "double_slash_count": url.count("//"),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "digit_ratio": (digit_count / length) if length else 0.0,
        "special_char_ratio": (special_char_count / length) if length else 0.0,
        "url_entropy": _entropy(url),
    }
    return features


ALL_FEATURE_COLUMNS = [
    "url_length", "dot_count", "hyphen_count", "special_char_count",
    "has_https", "subdomain_count", "has_ip", "suspicious_keyword_count",
    "digit_count", "letter_count", "slash_count", "question_mark_count",
    "equals_count", "ampersand_count", "percent_count", "at_symbol_count",
    "colon_count", "double_slash_count", "domain_length", "path_length",
    "query_length", "digit_ratio", "special_char_ratio", "url_entropy",
]

# has_https, colon_count and double_slash_count are excluded from the model
# feature vector: in feature_dataset_v2.csv, whether a URL includes a
# "http(s)://" scheme prefix turned out to be a near-perfect proxy for which
# source list it came from (43.6% of scam-labeled URLs had a scheme prefix
# vs. 8.0% of legitimate ones), not a genuine signal. double_slash_count
# alone accounted for 54% of a trained XGBoost model's feature importance
# and caused clearly-legitimate HTTPS sites (e.g. google.com) to be flagged
# as scams. These three stay in extract_features() for display purposes
# (has_https is still a useful reason to surface) but are dropped from
# FEATURE_COLUMNS so the models can't use them as a shortcut.
FEATURE_COLUMNS = [
    c for c in ALL_FEATURE_COLUMNS
    if c not in ("has_https", "colon_count", "double_slash_count")
]


def extract_feature_vector(url: str):
    feats = extract_features(url)
    return [feats[c] for c in FEATURE_COLUMNS]
