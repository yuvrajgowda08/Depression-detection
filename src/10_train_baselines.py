import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

# load split data
train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/val.csv")
test = pd.read_csv("data/splits/test.csv")

Xtr, ytr = train["text"].tolist(), train["label"].values
Xva, yva = val["text"].tolist(), val["label"].values
Xte, yte = test["text"].tolist(), test["label"].values

# ----------------------------
# TF-IDF + Logistic Regression
# ----------------------------

lr = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_df=1.0
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

lr.fit(Xtr, ytr)

val_pred = lr.predict(Xva)
test_pred = lr.predict(Xte)

print("TFIDF + LogisticRegression")
print("Validation F1:", f1_score(yva, val_pred))
print("Test F1:", f1_score(yte, test_pred))

joblib.dump(lr, "outputs/tfidf_lr.joblib")

# ----------------------------
# TF-IDF + Linear SVM
# ----------------------------

svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_df=1.0
    )),
    ("clf", LinearSVC(
        class_weight="balanced"
    ))
])

svm.fit(Xtr, ytr)

val_pred = svm.predict(Xva)
test_pred = svm.predict(Xte)

print("\nTFIDF + LinearSVM")
print("Validation F1:", f1_score(yva, val_pred))
print("Test F1:", f1_score(yte, test_pred))

joblib.dump(svm, "outputs/tfidf_svm.joblib")

print("\nModels saved to outputs/")