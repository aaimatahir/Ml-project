"""
Person 3 module: website analysis and NLP.

Given a URL, fetches the live page (best-effort, never raises) and inspects
its HTML structure -- forms, login/password fields, external links, contact
page, privacy policy, iframes, favicon origin -- and scores its visible text
for suspicious language using a small TF-IDF-weighted phishing lexicon.
Produces its own suspicion score and reasons, independent of the URL-string
and DL models, to be combined by Person 5's risk engine.
"""

import re
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

REQUEST_TIMEOUT = 6
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ScamShieldBot/1.0"
)

SUSPICIOUS_PHRASES = [
    "verify your account", "confirm your identity", "account has been suspended",
    "unusual activity", "click here immediately", "limited time offer",
    "update your payment", "your account will be locked", "provide your password",
    "confirm your password", "security alert", "act now", "claim your reward",
    "you have won", "urgent action required", "restore access",
    "log in to continue", "re-enter your password", "billing information",
    "suspicious login attempt", "unlock your account", "verify now",
    "your account is at risk", "failure to update", "confirm payment details",
]

_VECTORIZER = TfidfVectorizer(vocabulary=SUSPICIOUS_PHRASES, ngram_range=(1, 5))


def _text_suspicion_score(text: str):
    text = text.lower()
    if not text.strip():
        return 0.0, []
    try:
        tfidf = _VECTORIZER.fit_transform([text])
    except ValueError:
        return 0.0, []
    scores = tfidf.toarray()[0]
    hits = [(SUSPICIOUS_PHRASES[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
    hits.sort(key=lambda x: -x[1])
    score = min(1.0, sum(s for _, s in hits))
    return score, [phrase for phrase, _ in hits[:5]]


def fetch_page(url: str):
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        return resp, None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def analyze_website(url: str) -> dict:
    result = {
        "reachable": False,
        "error": None,
        "final_url": url,
        "status_code": None,
        "form_count": 0,
        "password_field_count": 0,
        "login_form_over_http": False,
        "iframe_count": 0,
        "external_link_count": 0,
        "internal_link_count": 0,
        "has_contact_page": False,
        "has_privacy_policy": False,
        "favicon_external": False,
        "text_suspicion_score": 0.0,
        "suspicious_phrases_found": [],
        "reasons": [],
        "suspicion_score": 0.0,
    }

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    resp, err = fetch_page(url)
    if resp is None:
        result["error"] = f"Could not reach site: {err}"
        result["reasons"].append("Website could not be reached for analysis")
        result["suspicion_score"] = 0.3
        return result

    result["reachable"] = True
    result["final_url"] = resp.url
    result["status_code"] = resp.status_code

    if resp.status_code >= 400:
        result["reasons"].append(f"Site returned HTTP {resp.status_code}")
        result["suspicion_score"] = 0.3

    parsed = urlparse(resp.url)
    page_domain = parsed.netloc.lower()

    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        soup = BeautifulSoup(resp.text, "html.parser")

    forms = soup.find_all("form")
    result["form_count"] = len(forms)
    password_fields = soup.find_all("input", {"type": "password"})
    result["password_field_count"] = len(password_fields)

    if password_fields and parsed.scheme != "https":
        result["login_form_over_http"] = True
        result["reasons"].append("Login form collects a password over plain HTTP")

    result["iframe_count"] = len(soup.find_all("iframe"))
    if result["iframe_count"] > 2:
        result["reasons"].append("Page embeds an unusually high number of iframes")

    external, internal = 0, 0
    contact_found, privacy_found = False, False
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True).lower()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        full = urljoin(resp.url, href)
        link_domain = urlparse(full).netloc.lower()
        if link_domain and link_domain != page_domain:
            external += 1
        else:
            internal += 1
        combined = (href + " " + text).lower()
        if "contact" in combined:
            contact_found = True
        if "privacy" in combined:
            privacy_found = True

    result["external_link_count"] = external
    result["internal_link_count"] = internal
    result["has_contact_page"] = contact_found
    result["has_privacy_policy"] = privacy_found

    if not privacy_found:
        result["reasons"].append("No privacy policy link found")
    if not contact_found:
        result["reasons"].append("No contact page link found")

    icon = soup.find("link", rel=lambda v: v and "icon" in v.lower())
    if icon and icon.get("href"):
        icon_domain = urlparse(urljoin(resp.url, icon["href"])).netloc.lower()
        if icon_domain and icon_domain != page_domain:
            result["favicon_external"] = True
            result["reasons"].append("Favicon is served from a different domain")

    page_text = soup.get_text(" ", strip=True)
    text_score, phrases = _text_suspicion_score(page_text)
    result["text_suspicion_score"] = text_score
    result["suspicious_phrases_found"] = phrases
    if phrases:
        result["reasons"].append(
            "Suspicious phrasing detected: " + ", ".join(phrases)
        )

    score = 0.0
    if result["login_form_over_http"]:
        score += 0.35
    if not privacy_found:
        score += 0.1
    if not contact_found:
        score += 0.05
    if result["favicon_external"]:
        score += 0.15
    if result["iframe_count"] > 2:
        score += 0.1
    score += min(0.4, text_score)

    result["suspicion_score"] = round(min(1.0, score), 4)
    return result
