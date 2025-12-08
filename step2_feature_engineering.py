"""
Step 2: Feature Engineering
Creates lag features, rolling statistics, and temporal features for ML
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

# Load cleaned data
print("\n📥 Loading cleaned transaction data...")
df = pd.read_csv('cleaned_data.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"   Loaded {len(df):,} records")

# Aggregate daily sales by store and SKU
print("\n📊 Aggregating daily sales...")
daily_sales = df.groupby(['date', 'store_id', 'sku_id']).agg({
    'qty_sold': 'sum',
    'transaction_price': 'mean',
    'revenue': 'sum',
    'product_name': 'first',
    'category': 'first'
}).reset_index()
daily_sales = daily_sales.sort_values('date').reset_index(drop=True)
print(f"   Created {len(daily_sales):,} daily aggregations")

# Create complete date range for each store-SKU combination
print("\n🗓️  Creating complete date range...")
date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
store_sku_combos = daily_sales[['store_id', 'sku_id']].drop_duplicates()

# Create full matrix
full_matrix = []
for _, row in store_sku_combos.iterrows():
    for date in date_range:
        full_matrix.append({
            'date': date,
            'store_id': row['store_id'],
            'sku_id': row['sku_id']
        })

df_full = pd.DataFrame(full_matrix)
print(f"   Full matrix: {len(df_full):,} records ({df_full['store_id'].nunique()} stores × {df_full['sku_id'].nunique()} SKUs × {len(date_range)} days)")

# Merge with actual sales
df_features = df_full.merge(daily_sales, on=['date', 'store_id', 'sku_id'], how='left')

# Fill missing values (zero sales days)
df_features['qty_sold'] = df_features['qty_sold'].fillna(0)
df_features['revenue'] = df_features['revenue'].fillna(0)
df_features['transaction_price'] = df_features['transaction_price'].fillna(0)

# Forward fill product info
df_features = df_features.sort_values('date')
for col in ['product_name', 'category']:
    df_features[col] = df_features.groupby(['store_id', 'sku_id'])[col].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )

print(f"   After filling: {len(df_features):,} records with zero-filled no-sale days")

# LAG FEATURES (Previous N days' sales)
print("\n⏱️  Creating lag features...")
lag_periods = [1, 7, 14]
for lag in lag_periods:
    df_features[f'qty_lag_{lag}d'] = df_features.groupby(['store_id', 'sku_id'])['qty_sold'].shift(lag)
print(f"   ✓ Added {len(lag_periods)} lag features")

# ROLLING STATISTICS
print("\n📈 Creating rolling statistics...")
rolling_windows = [7, 14, 30]
for window in rolling_windows:
    # Rolling mean
    df_features[f'qty_rolling_mean_{window}d'] = df_features.groupby(['store_id', 'sku_id'])['qty_sold'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    # Rolling std dev
    df_features[f'qty_rolling_std_{window}d'] = df_features.groupby(['store_id', 'sku_id'])['qty_sold'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    ).fillna(0)

print(f"   ✓ Added {len(rolling_windows) * 2} rolling features")

# TEMPORAL FEATURES
print("\n🕐 Creating temporal features...")
df_features['day_of_week'] = df_features['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df_features['day_of_month'] = df_features['date'].dt.day
df_features['month'] = df_features['date'].dt.month
df_features['quarter'] = df_features['date'].dt.quarter
df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
df_features['day_of_year'] = df_features['date'].dt.dayofyear

# Cyclical encoding for day of week and month
df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

# Weekend flag
df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

print(f"   ✓ Added 11 temporal features")

# CATEGORICAL ENCODING
print("\n🏷️  Encoding categorical features...")
df_features['store_id_encoded'] = pd.factorize(df_features['store_id'])[0]
df_features['sku_id_encoded'] = pd.factorize(df_features['sku_id'])[0]

# One-hot encode category
category_dummies = pd.get_dummies(df_features['category'], prefix='category')
df_features = pd.concat([df_features, category_dummies], axis=1)
print(f"   ✓ Encoded categorical features")

# Handle NaN values from lag features
print("\n🔧 Finalizing features...")
df_features = df_features.fillna(method='bfill').fillna(method='ffill').fillna(0)
print(f"   ✓ Handled missing values")

# Feature summary
print("\n✅ Feature Engineering Summary:")
print(f"   Total records: {len(df_features):,}")
print(f"   Total features: {len(df_features.columns)}")
print(f"   Date range: {df_features['date'].min().date()} to {df_features['date'].max().date()}")
print(f"   Store-SKU combinations: {df_features.groupby(['store_id', 'sku_id']).ngroups}")

# List all features
feature_cols = [col for col in df_features.columns if col not in ['date', 'product_name', 'category']]
print(f"\n📋 Feature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

# Statistics
print("\n📊 Feature Statistics:")
print(f"   Avg daily qty_sold: {df_features['qty_sold'].mean():.2f} units")
print(f"   Max daily qty_sold: {df_features['qty_sold'].max():.0f} units")
print(f"   Days with zero sales: {(df_features['qty_sold'] == 0).sum():,} ({(df_features['qty_sold'] == 0).sum()/len(df_features)*100:.1f}%)")

# Save engineered features
output_file = 'features_engineered_data.csv'
df_features.to_csv(output_file, index=False)
print(f"\n💾 Engineered features saved to: {output_file}")
print(f"   Shape: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")

# Display sample
print("\n📋 Sample features (first 5 rows, key columns):")
sample_cols = ['date', 'store_id', 'sku_id', 'qty_sold', 'qty_lag_1d', 'qty_lag_7d',
               'qty_rolling_mean_7d', 'qty_rolling_mean_30d', 'day_of_week', 'is_weekend', 'month']
print(df_features[sample_cols].head().to_string(index=False))

print("\n" + "=" * 60)
print("✅ STEP 2 COMPLETE")
print("=" * 60)
