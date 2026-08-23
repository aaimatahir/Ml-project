"""
Person 5 module: combines URL analysis (Person 1), ML prediction (Person 2),
website/NLP analysis (Person 3) and DL prediction (Person 4) into one final
risk score, verdict, and list of reasons.
"""

from urllib.parse import urlparse

import numpy as np
import joblib

from feature_extraction import extract_features, extract_feature_vector
import ml_predict
import dl_predict
from website_analysis import analyze_website
from known_legitimate_domains import KNOWN_LEGITIMATE_DOMAINS

_ALLOWLIST = set(KNOWN_LEGITIMATE_DOMAINS)


def _is_allowlisted(url: str) -> bool:
    host = urlparse(url if "://" in url else f"http://{url}").netloc.lower()
    host = host.split(":")[0]
    # match the host itself or any of its parent domains, so subdomains of
    # an allowlisted brand (pk.indeed.com, mail.google.com, ...) also match
    labels = host.split(".")
    for i in range(len(labels) - 1):
        if ".".join(labels[i:]) in _ALLOWLIST:
            return True
    return False

SCALER_PATH = "models/scaler.pkl"

_scaler_cache = None


def _get_scaler():
    global _scaler_cache
    if _scaler_cache is None:
        _scaler_cache = joblib.load(SCALER_PATH)
    return _scaler_cache


def _url_reasons(features: dict):
    reasons = []
    if features["has_ip"]:
        reasons.append("URL uses a raw IP address instead of a domain name")
    if not features["has_https"]:
        reasons.append("URL does not use HTTPS")
    if features["suspicious_keyword_count"] > 0:
        reasons.append(f"URL contains {features['suspicious_keyword_count']} suspicious keyword(s)")
    if features["subdomain_count"] >= 3:
        reasons.append(f"URL has an unusually high number of subdomains ({features['subdomain_count']})")
    if features["at_symbol_count"] > 0:
        reasons.append("URL contains an '@' symbol, often used to obscure the real destination")
    if features["url_length"] > 75:
        reasons.append("URL is unusually long")
    if features["url_entropy"] > 4.5:
        reasons.append("URL has unusually high randomness (possibly auto-generated)")
    return reasons


def _verdict(risk: float):
    if risk >= 0.65:
        return "Scam"
    if risk >= 0.35:
        return "Suspicious"
    return "Safe"


def assess_url(url: str, run_website_analysis: bool = True) -> dict:
    url = url.strip()

    features = extract_features(url)
    vector = np.array(extract_feature_vector(url), dtype=np.float64).reshape(1, -1)
    scaled = _get_scaler().transform(vector)

    ml_prob, ml_model_name = ml_predict.predict_scam_probability(scaled)
    dl_prob = dl_predict.predict_scam_probability(scaled)

    reasons = _url_reasons(features)

    website = None
    web_prob = None
    if run_website_analysis:
        website = analyze_website(url)
        web_prob = website["suspicion_score"]
        reasons.extend(website.get("reasons", []))

    if web_prob is not None:
        risk = 0.35 * ml_prob + 0.35 * dl_prob + 0.30 * web_prob
    else:
        risk = 0.5 * ml_prob + 0.5 * dl_prob

    risk = float(round(min(1.0, max(0.0, risk)), 4))

    if _is_allowlisted(url):
        risk = min(risk, 0.05)
        reasons.insert(0, "Domain matches a known, trusted allowlist")

    verdict = _verdict(risk)

    if not reasons:
        reasons.append("No obvious red flags detected in URL structure or page content")

    return {
        "url": url,
        "verdict": verdict,
        "risk_percent": round(risk * 100, 1),
        "ml_probability": round(ml_prob * 100, 1),
        "ml_model": ml_model_name,
        "dl_probability": round(dl_prob * 100, 1),
        "website_probability": round(web_prob * 100, 1) if web_prob is not None else None,
        "reasons": reasons,
        "features": features,
        "website": website,
    }
