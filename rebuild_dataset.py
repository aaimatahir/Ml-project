"""
Regenerates the 24 feature columns for every row in feature_dataset_v2.csv
directly from the 'url' column using feature_extraction.py, so training
data and the live Flask predictor use identical feature computation.
"""

import time
import numpy as np
import pandas as pd

from feature_extraction import extract_feature_vector, FEATURE_COLUMNS

print("Loading raw dataset...")
df = pd.read_csv("feature_dataset_v2.csv")
df = df.dropna(subset=["url", "status"])
print("Rows:", len(df))

t0 = time.time()
vectors = np.empty((len(df), len(FEATURE_COLUMNS)), dtype=np.float64)
urls = df["url"].astype(str).tolist()

for i, u in enumerate(urls):
    vectors[i] = extract_feature_vector(u)
    if i % 100000 == 0:
        print(f"  {i}/{len(urls)} ({time.time()-t0:.1f}s)")

print(f"Done in {time.time()-t0:.1f}s")

out = pd.DataFrame(vectors, columns=FEATURE_COLUMNS)
out.insert(0, "url", df["url"].values)
out["status"] = df["status"].values

out.to_csv("feature_dataset_v2_regenerated.csv", index=False)
print("Saved feature_dataset_v2_regenerated.csv", out.shape)
