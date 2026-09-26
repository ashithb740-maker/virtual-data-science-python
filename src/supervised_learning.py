import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv(
    "data/processed/online_retail_cleaned.csv.gz",
    compression="gzip"
)

customer = df.groupby("CustomerID").agg(
    TotalSpending=("TotalAmount", "sum"),
    NumberOfOrders=("InvoiceNo", "nunique"),
    AverageOrderValue=("TotalAmount", "mean"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfProducts=("Description", "nunique")
).reset_index()

# Target: top 25% of customers by total spending
threshold = customer["TotalSpending"].quantile(0.75)
customer["HighValue"] = (
    customer["TotalSpending"] >= threshold
).astype(int)

features = [
    "NumberOfOrders",
    "AverageOrderValue",
    "TotalQuantity",
    "NumberOfProducts"
]

X = customer[features]
y = customer["HighValue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
print("5-fold F1 scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())
