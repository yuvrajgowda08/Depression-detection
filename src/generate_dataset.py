import pandas as pd
import random
from datetime import datetime, timedelta

depressed_texts = [
    "I feel very sad and lonely",
    "Life feels meaningless",
    "I have no motivation anymore",
    "I feel empty inside",
    "Nothing makes me happy anymore",
    "I feel hopeless about my future",
    "I feel tired of everything",
    "I feel like nobody understands me",
    "I feel very lonely these days",
    "I can't find joy in anything"
]

normal_texts = [
    "I had a great day today",
    "I enjoyed spending time with friends",
    "Today was productive and fun",
    "I love playing sports",
    "I watched a great movie today",
    "I feel happy and motivated",
    "I enjoyed a nice walk outside",
    "I had a good conversation today",
    "I am excited about tomorrow",
    "Today was a relaxing day"
]

rows = []
start_date = datetime(2024, 1, 1)

for i in range(10000):

    label = random.randint(0,1)

    if label == 1:
        text = random.choice(depressed_texts)
    else:
        text = random.choice(normal_texts)

    date = start_date + timedelta(days=random.randint(0,365))

    rows.append({
        "user_id": random.randint(1,2000),
        "text": text,
        "timestamp": date.strftime("%Y-%m-%d"),
        "label": label
    })

df = pd.DataFrame(rows)

df.to_csv("data/raw/posts.csv", index=False)

print("Generated 10,000 rows in data/raw/posts.csv")