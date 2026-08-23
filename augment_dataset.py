"""
Appends bare/https/www root-URL examples of well-known legitimate domains
(status=1) to the regenerated dataset, to patch the sampling gap where the
"legitimate" class barely contained any bare-root URLs. See
known_legitimate_domains.py for why.
"""

import pandas as pd

from feature_extraction import extract_features, FEATURE_COLUMNS
from known_legitimate_domains import KNOWN_LEGITIMATE_DOMAINS

df = pd.read_csv("feature_dataset_v2_regenerated.csv")
print("Before augmentation:", df.shape)

rows = []
for domain in KNOWN_LEGITIMATE_DOMAINS:
    for url in (domain, f"https://{domain}", f"https://www.{domain}", f"https://{domain}/"):
        feats = extract_features(url)
        row = {"url": url, **{c: feats[c] for c in FEATURE_COLUMNS}, "status": 1}
        rows.append(row)

# Replicated (not just added once) so this pattern has enough weight for the
# tree-based / gradient-boosted models to actually carve out a region for it,
# rather than being statistically swamped by 565k+ original rows.
REPLICAS = 15
aug_df = pd.DataFrame(rows * REPLICAS)
print("Augmentation rows (with replicas):", aug_df.shape)

out = pd.concat([df, aug_df], ignore_index=True)
out.to_csv("feature_dataset_v2_augmented.csv", index=False)
print("Saved feature_dataset_v2_augmented.csv", out.shape)
