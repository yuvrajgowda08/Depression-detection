from utils import load_posts, basic_clean, ensure_dir

ensure_dir("data/processed")

df = load_posts("data/raw/posts.csv")
df["text_clean"] = df["text"].apply(basic_clean)

df = df[df["text_clean"].str.len() > 0].copy()

out = "data/processed/posts_clean.csv"
df.to_csv(out, index=False)
print("Saved:", out, "rows:", len(df))
