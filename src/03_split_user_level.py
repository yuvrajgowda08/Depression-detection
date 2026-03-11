import pandas as pd
from utils import user_level_split, ensure_dir

ensure_dir("data/splits")

df = pd.read_csv("data/processed/user_windows.csv")

train, val, test = user_level_split(df, test_size=0.2, val_size=0.1, seed=42)

train.to_csv("data/splits/train.csv", index=False)
val.to_csv("data/splits/val.csv", index=False)
test.to_csv("data/splits/test.csv", index=False)

print("Train users:", train["user_id"].nunique(), "rows:", len(train))
print("Val users:", val["user_id"].nunique(), "rows:", len(val))
print("Test users:", test["user_id"].nunique(), "rows:", len(test))