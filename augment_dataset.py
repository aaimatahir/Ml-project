"""
Patches a sampling gap in feature_dataset_v2.csv: the "legitimate" class was
sourced almost entirely from deep content links (median path length 22
chars) and barely contains bare-root URLs, while the "scam" class is
disproportionately bare-root (54% vs 12%). A model trained as-is learns
"URL has little structure => scam" and flags ordinary sites -- including
ones with a scheme, a subdomain, or a simple query string -- as scams.

Two augmentation passes, both using sample weights (via a `sample_weight`
column carried through preprocessing/training) rather than physically
duplicating rows -- this lets the strength of the correction be tuned by
changing a constant and retraining, without regenerating gigabytes of data
each time, and lets pass 2 cover every known-legitimate domain at full
shape diversity instead of a small sample:

1. A curated list of ~130 well-known brand domains (also used for the
   risk_engine allowlist safety net).
2. Every unique domain already labeled legitimate (status==1) somewhere in
   the real dataset, rendered across realistic shapes (bare, https, https
   with trailing slash, https with a simple query string) -- teaching the
   *general* pattern that scheme/subdomain/query presence isn't inherently
   suspicious, across ~146k genuinely-legitimate domains.
"""

import pandas as pd

from feature_extraction import extract_features, FEATURE_COLUMNS, _split_domain_path_query
from known_legitimate_domains import KNOWN_LEGITIMATE_DOMAINS

BRAND_WEIGHT = 15
SCALE_WEIGHT = 15

df = pd.read_csv("feature_dataset_v2_regenerated.csv")
df["sample_weight"] = 1.0
print("Before augmentation:", df.shape)

# --- Pass 1: curated brand list ---
rows = []
for domain in KNOWN_LEGITIMATE_DOMAINS:
    for url in (domain, f"https://{domain}", f"https://www.{domain}", f"https://{domain}/"):
        feats = extract_features(url)
        row = {"url": url, **{c: feats[c] for c in FEATURE_COLUMNS}, "status": 1,
               "sample_weight": BRAND_WEIGHT}
        rows.append(row)

brand_df = pd.DataFrame(rows)
print("Brand augmentation rows:", brand_df.shape)

# --- Pass 2: every unique legitimate domain, across realistic URL shapes ---
legit = df[df.status == 1]
domains = legit["url"].astype(str).apply(lambda u: _split_domain_path_query(u)[0])
unique_domains = domains[domains.str.contains(r"\.", regex=True) & (domains.str.len() > 3)].unique()
print("Unique legitimate domains found in dataset:", len(unique_domains))

scale_rows = []
for domain in unique_domains:
    for url in (domain, f"https://{domain}", f"https://{domain}/", f"https://{domain}/?ref=1"):
        feats = extract_features(url)
        scale_rows.append({"url": url, **{c: feats[c] for c in FEATURE_COLUMNS}, "status": 1,
                            "sample_weight": SCALE_WEIGHT})

scale_df = pd.DataFrame(scale_rows)
print("Large-scale multi-shape augmentation rows:", scale_df.shape)

out = pd.concat([df, brand_df, scale_df], ignore_index=True)
out.to_csv("feature_dataset_v2_augmented.csv", index=False)
print("Saved feature_dataset_v2_augmented.csv", out.shape)
print("New status balance (unweighted row counts):")
print(out["status"].value_counts())
print("Weighted mass by status:")
print(out.groupby("status")["sample_weight"].sum())
