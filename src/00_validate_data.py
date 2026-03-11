from utils import load_posts

df = load_posts("data/raw/posts.csv")
print(df.head())
print("\nRows:", len(df))
print("Users:", df["user_id"].nunique())
print("Label balance:\n", df["label"].value_counts(normalize=True))
print("Domains:\n", df["domain"].value_counts().head(10))