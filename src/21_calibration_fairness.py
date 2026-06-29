import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss

def ece(probs, y, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    e = 0.0
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m = (probs>=lo)&(probs<hi)
        if m.sum()==0: 
            continue
        acc = y[m].mean()
        conf = probs[m].mean()
        e += (m.mean())*abs(acc-conf)
    return float(e)

df = pd.read_csv("data/splits/test.csv")
model = joblib.load("outputs/tfidf_lr.joblib")

probs = model.predict_proba(df["text"].tolist())[:,1]
y = df["label"].values

print("ECE:", ece(probs,y))
print("Brier:", brier_score_loss(y, probs))

if "n_posts" in df.columns:
    user_activity = df.groupby("user_id")["n_posts"].sum()
    q1,q2 = user_activity.quantile([0.33,0.66]).values
    strata = {}
    for uid,v in user_activity.items():
        if v<=q1: strata[uid]="low"
        elif v<=q2: strata[uid]="mid"
        else: strata[uid]="high"
    df["activity"] = df["user_id"].map(strata)

    pred = (probs>=0.5).astype(int)
    from sklearn.metrics import f1_score
    for g in ["low","mid","high"]:
        m = (df["activity"]==g)
        if m.sum()>0:
            print(g, "F1:", f1_score(y[m], pred[m]), "ECE:", ece(probs[m], y[m]))
else:
    print("No n_posts column found; add it from user_windows.csv if you want activity strata.")
