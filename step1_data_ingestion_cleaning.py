"""
Step 1: Data Ingestion & Cleaning
Loads raw transaction data and prepares it for feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: DATA INGESTION & CLEANING")
print("=" * 60)

# Load raw transaction data
print("\n📥 Loading raw transaction data...")
df = pd.read_csv('historical_transactions.csv')
initial_rows = len(df)
print(f"   Loaded {initial_rows:,} transaction records")

# Load product hierarchy for enrichment
products_df = pd.read_csv('product_hierarchy.csv')
print(f"   Loaded {len(products_df)} products")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Data cleaning operations
print("\n🧹 Cleaning data...")

# 1. Remove duplicates
duplicates_before = len(df)
df = df.drop_duplicates(subset=['date', 'store_id', 'sku_id'])
duplicates_removed = duplicates_before - len(df)
print(f"   ✓ Removed {duplicates_removed} duplicate records")

# 2. Handle missing values
missing_before = df.isnull().sum().sum()
df = df.dropna()
missing_removed = missing_before - df.isnull().sum().sum()
print(f"   ✓ Removed {missing_removed} rows with missing values")

# 3. Validate numeric columns
df['qty_sold'] = pd.to_numeric(df['qty_sold'], errors='coerce')
df['transaction_price'] = pd.to_numeric(
    df['transaction_price'], errors='coerce')
df = df.dropna(subset=['qty_sold', 'transaction_price'])
print(f"   ✓ Validated numeric columns (qty_sold, transaction_price)")

# 4. Remove outliers (extreme prices or quantities)
Q1_qty = df['qty_sold'].quantile(0.25)
Q3_qty = df['qty_sold'].quantile(0.75)
IQR_qty = Q3_qty - Q1_qty
qty_lower = Q1_qty - 1.5 * IQR_qty
qty_upper = Q3_qty + 1.5 * IQR_qty

Q1_price = df['transaction_price'].quantile(0.25)
Q3_price = df['transaction_price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
price_lower = Q1_price - 1.5 * IQR_price
price_upper = Q3_price + 1.5 * IQR_price

outliers_before = len(df)
df = df[
    (df['qty_sold'] >= qty_lower) & (df['qty_sold'] <= qty_upper) &
    (df['transaction_price'] >= price_lower) & (
        df['transaction_price'] <= price_upper)
]
outliers_removed = outliers_before - len(df)
print(f"   ✓ Removed {outliers_removed} outlier records")

# 5. Ensure store_id and sku_id are valid integers
df['store_id'] = df['store_id'].astype(int)
df['sku_id'] = df['sku_id'].astype(int)
print(f"   ✓ Validated store and SKU IDs")

# 6. Sort by date
df = df.sort_values('date').reset_index(drop=True)
print(f"   ✓ Sorted data chronologically")

# Enrich with product information
print("\n📊 Enriching with product metadata...")
df = df.merge(
    products_df[['sku_id', 'product_name',
                 'category', 'cost_price', 'sell_price']],
    on='sku_id',
    how='left'
)
print(f"   ✓ Added product name, category, and pricing")
# Consolidate possible duplicate columns if the input already included product fields
if 'cost_price' not in df.columns:
    # Handle pandas suffixes from double-merge (e.g., cost_price_x / cost_price_y)
    if 'cost_price_x' in df.columns or 'cost_price_y' in df.columns:
        df['cost_price'] = df.get('cost_price_x').fillna(
            df.get('cost_price_y'))
    if 'sell_price_x' in df.columns or 'sell_price_y' in df.columns:
        df['sell_price'] = df.get('sell_price_x').fillna(
            df.get('sell_price_y'))
    if 'product_name_x' in df.columns or 'product_name_y' in df.columns:
        df['product_name'] = df.get('product_name_x').fillna(
            df.get('product_name_y'))
    if 'category_x' in df.columns or 'category_y' in df.columns:
        df['category'] = df.get('category_x').fillna(df.get('category_y'))
    # Drop the old suffixed columns if they exist
    for col in ['cost_price_x', 'cost_price_y', 'sell_price_x', 'sell_price_y', 'product_name_x', 'product_name_y', 'category_x', 'category_y']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

# Calculate additional metrics
df['revenue'] = df['qty_sold'] * df['transaction_price']
df['margin'] = (df['transaction_price'] - df['cost_price']) * df['qty_sold']
df['margin_pct'] = (
    (df['transaction_price'] - df['cost_price']) / df['cost_price'] * 100).round(2)
print(f"   ✓ Calculated revenue and margin metrics")

# Data quality checks
print("\n✅ Data Quality Summary:")
print(f"   Total records: {len(df):,} (from {initial_rows:,})")
print(
    f"   Total records removed: {initial_rows - len(df):,} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"   Unique stores: {df['store_id'].nunique()}")
print(f"   Unique SKUs: {df['sku_id'].nunique()}")
print(f"   Unique products: {df['product_name'].nunique()}")
print(f"   Categories: {df['category'].nunique()}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Statistics
print("\n📈 Transaction Statistics:")
print(f"   Avg quantity per transaction: {df['qty_sold'].mean():.2f} units")
print(f"   Avg price per unit: ${df['transaction_price'].mean():.2f}")
print(f"   Total revenue: ${df['revenue'].sum():,.2f}")
print(f"   Total margin: ${df['margin'].sum():,.2f}")
print(f"   Avg margin %: {df['margin_pct'].mean():.1f}%")

# Show category breakdown
print("\n📦 Revenue by Category:")
category_revenue = df.groupby(
    'category')['revenue'].sum().sort_values(ascending=False)
for category, revenue in category_revenue.items():
    pct = revenue / df['revenue'].sum() * 100
    print(f"   {category}: ${revenue:,.2f} ({pct:.1f}%)")

# Show store performance
print("\n🏪 Revenue by Store:")
store_revenue = df.groupby('store_id')[
    'revenue'].sum().sort_values(ascending=False)
for store_id, revenue in store_revenue.items():
    pct = revenue / df['revenue'].sum() * 100
    print(f"   Store {store_id}: ${revenue:,.2f} ({pct:.1f}%)")

# Save cleaned data
output_file = 'cleaned_data.csv'
df.to_csv(output_file, index=False)
print(f"\n💾 Cleaned data saved to: {output_file}")
print(f"   Columns: {len(df.columns)}")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Display sample
print("\n📋 Sample of cleaned data (first 5 rows):")
print(df[['date', 'store_id', 'product_name', 'qty_sold', 'transaction_price',
      'revenue', 'margin_pct']].head().to_string(index=False))

print("\n" + "=" * 60)
print("✅ STEP 1 COMPLETE")
print("=" * 60)
