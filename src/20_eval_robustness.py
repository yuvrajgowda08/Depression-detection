import pandas as pd
import joblib
from sklearn.metrics import f1_score

df = pd.read_csv("data/processed/user_windows.csv")

DOMAIN_A = "domainA"
DOMAIN_B = "domainB"

train = df[df["domain"]==DOMAIN_A].copy()
test  = df[df["domain"]==DOMAIN_B].copy()

overlap = set(train["user_id"]).intersection(set(test["user_id"]))
if overlap:
    test = test[~test["user_id"].isin(overlap)].copy()

model = joblib.load("outputs/tfidf_lr.joblib")
pred = model.predict(test["text"].tolist())
print("Cross-domain F1:", f1_score(test["label"].values, pred))
