import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/processed/online_retail_cleaned.csv.gz", compression="gzip")

customer = df.groupby("CustomerID").agg(
    TotalSpending=("TotalAmount", "sum"),
    NumberOfOrders=("InvoiceNo", "nunique"),
    AverageOrderValue=("TotalAmount", "mean"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfProducts=("Description", "nunique")
).reset_index()

features = ['TotalSpending', 'NumberOfOrders', 'AverageOrderValue', 'TotalQuantity', 'NumberOfProducts']
X = np.log1p(customer[features])
X_scaled = StandardScaler().fit_transform(X)

# Compare k values using inertia and silhouette score
results = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    results.append((k, model.inertia_, silhouette_score(X_scaled, labels)))

# Selected k based on the highest silhouette score
best_k = max(results, key=lambda x: x[2])[0]
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
customer["Cluster"] = model.fit_predict(X_scaled)

customer.to_csv("data/processed/customer_clusters.csv", index=False)
customer.groupby("Cluster")[features].agg(["count", "mean", "median"]).to_csv(
    "data/processed/customer_cluster_profiles.csv"
)

print("Selected clusters:", best_k)
print("Cluster sizes:")
print(customer["Cluster"].value_counts().sort_index())
