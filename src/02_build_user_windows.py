import pandas as pd
from utils import ensure_dir

ensure_dir("data/processed")

df = pd.read_csv("data/processed/posts_clean.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Choose window size (e.g., 14 days)
WINDOW_DAYS = 14

# bucket by window index per user
df = df.sort_values(["user_id", "timestamp"])
df["t0"] = df.groupby("user_id")["timestamp"].transform("min")
df["window_id"] = ((df["timestamp"] - df["t0"]).dt.days // WINDOW_DAYS).astype(int)

# aggregate text per (user, window)
agg = (
    df.groupby(["user_id", "window_id", "domain", "label"], as_index=False)
      .agg(
          window_start=("timestamp", "min"),
          window_end=("timestamp", "max"),
          text=("text_clean", lambda x: " </s> ".join(x.tolist())),
          n_posts=("text_clean", "size")
      )
)

out = "data/processed/user_windows.csv"
agg.to_csv(out, index=False)
print("Saved:", out, "rows:", len(agg), "users:", agg["user_id"].nunique())