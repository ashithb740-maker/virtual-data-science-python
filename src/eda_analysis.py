import pandas as pd
import matplotlib.pyplot as plt

DATA = "data/processed/online_retail_cleaned.csv.gz"
FIGURES = "figures/"

# Load cleaned Week 1 dataset
df = pd.read_csv(DATA, compression="gzip")

# Prepare data types
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")

# -------------------------
# Initial EDA
# -------------------------
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDescriptive statistics:\n", df[["Quantity", "UnitPrice", "TotalAmount"]].describe())

# -------------------------
# 1. Monthly sales trend
# -------------------------
monthly_sales = (
    df.set_index("InvoiceDate")
      .resample("M")["TotalAmount"]
      .sum()
)

plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES + "05_monthly_sales_trend.png", dpi=200)
plt.close()

# -------------------------
# 2. Top products by revenue
# -------------------------
top_products = (
    df.groupby("Description")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .sort_values()
)

plt.figure(figsize=(10, 6))
plt.barh(top_products.index, top_products.values)
plt.title("Top 10 Products by Total Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(FIGURES + "06_top_products.png", dpi=200)
plt.close()

# -------------------------
# 3. Country sales
# -------------------------
country_sales = (
    df.groupby("Country")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .sort_values()
)

plt.figure(figsize=(10, 6))
plt.barh(country_sales.index, country_sales.values)
plt.title("Top 10 Countries by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(FIGURES + "07_country_sales.png", dpi=200)
plt.close()

# -------------------------
# 4. Quantity distribution
# -------------------------
quantity = df["Quantity"].dropna()
quantity = quantity[quantity <= quantity.quantile(0.99)]

plt.figure(figsize=(10, 6))
plt.hist(quantity, bins=40)
plt.title("Quantity Distribution (Capped at 99th Percentile)")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES + "08_quantity_distribution.png", dpi=200)
plt.close()

# -------------------------
# 5. Unit price distribution
# -------------------------
price = df["UnitPrice"].dropna()
price = price[price <= price.quantile(0.99)]

plt.figure(figsize=(10, 6))
plt.hist(price, bins=40)
plt.title("Unit Price Distribution (Capped at 99th Percentile)")
plt.xlabel("Unit Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES + "09_unitprice_distribution.png", dpi=200)
plt.close()

# -------------------------
# 6. Correlation analysis
# -------------------------
corr = df[["Quantity", "UnitPrice", "TotalAmount"]].corr()

plt.figure(figsize=(7, 5))
plt.imshow(corr, interpolation="nearest", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center")

plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig(FIGURES + "10_correlation_heatmap.png", dpi=200)
plt.close()

# -------------------------
# 7. Customer spending
# -------------------------
customer_spending = (
    df.groupby("CustomerID")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .sort_values()
)

plt.figure(figsize=(10, 6))
plt.barh(customer_spending.index.astype(str),
         customer_spending.values)
plt.title("Top 10 Customers by Total Spending")
plt.xlabel("Total Spending")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig(FIGURES + "11_customer_spending.png", dpi=200)
plt.close()

print("\nEDA completed successfully.")
print("Seven visualizations saved in:", FIGURES)
