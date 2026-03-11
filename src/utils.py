import os
import re
import emoji
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    # keep emojis but normalize them to text tokens
    t = emoji.demojize(t, delimiters=(" ", " "))
    t = URL_RE.sub(" <URL> ", t)
    t = MENTION_RE.sub(" <USER> ", t)
    t = t.replace("\n", " ")
    t = WS_RE.sub(" ", t).strip()
    return t

def load_posts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "text", "timestamp", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required={required}")
    if "domain" not in df.columns:
        df["domain"] = "unknown"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamps could not be parsed. Fix timestamp format.")
    df["label"] = df["label"].astype(int)
    return df

def user_level_split(df, test_size=0.2, val_size=0.1, seed=42):
    # user-level split to prevent leakage (paper requirement) :contentReference[oaicite:2]{index=2}
    users = df[["user_id", "label"]].drop_duplicates()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(users, groups=users["user_id"]))
    train_val_users = users.iloc[train_val_idx]["user_id"]
    test_users = users.iloc[test_idx]["user_id"]

    train_val = df[df["user_id"].isin(train_val_users)].copy()
    test = df[df["user_id"].isin(test_users)].copy()

    # split train into train/val at user-level
    users_tv = train_val[["user_id"]].drop_duplicates()
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed)
    tr_idx, va_idx = next(gss2.split(users_tv, groups=users_tv["user_id"]))
    tr_users = users_tv.iloc[tr_idx]["user_id"]
    va_users = users_tv.iloc[va_idx]["user_id"]

    train = train_val[train_val["user_id"].isin(tr_users)].copy()
    val = train_val[train_val["user_id"].isin(va_users)].copy()
    return train, val, test