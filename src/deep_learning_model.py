import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data/processed/customer_supervised_modeling.csv")
features = ["NumberOfOrders","AverageOrderValue","TotalQuantity","NumberOfProducts"]
X = df[features].values.astype("float32")
y = df["HighValue"].values.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train).view(-1,1)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test).view(-1,1)

model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(), nn.Dropout(0.20),
    nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.20),
    nn.Linear(16, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    probabilities = torch.sigmoid(model(X_test)).numpy().ravel()
predictions = (probabilities >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
