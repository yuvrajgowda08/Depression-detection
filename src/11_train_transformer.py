import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Baseline interpretability: top LR features :contentReference[oaicite:17]{index=17}
lr = joblib.load("outputs/tfidf_lr.joblib")
tfidf: TfidfVectorizer = lr.named_steps["tfidf"]
clf = lr.named_steps["clf"]

feat = np.array(tfidf.get_feature_names_out())
w = clf.coef_[0]

top_pos = feat[np.argsort(-w)[:30]]
top_neg = feat[np.argsort(w)[:30]]

print("\nTop depression-indicative features:\n", top_pos)
print("\nTop control-indicative features:\n", top_neg)