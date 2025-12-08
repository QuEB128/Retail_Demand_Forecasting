import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime

print("🔄 Expanding dataset with synthetic historical data...")

# Load product hierarchy
products_df = pd.read_csv('product_hierarchy.csv')

# Create 3 years of historical data: 2022-01-01 to 2024-12-31
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2024-12-31')

stores = [1, 2, 3]
skus = products_df['sku_id'].values

expanded_transactions = []
current_date = start_date

while current_date <= end_date:
    for store_id in stores:
        for sku_id in skus:
            # Get product info
            product = products_df[products_df['sku_id'] == sku_id].iloc[0]

            # Create realistic demand patterns
            day_of_week = current_date.dayofweek
            month = current_date.month

            # Base demand
            base_qty = np.random.normal(35, 8)

            # Weekend boost
            weekend_multiplier = 1.4 if day_of_week >= 5 else 1.0

            # Seasonal variation
            if month == 12:
                seasonal_multiplier = 1.5
            elif month in [6, 7]:
                seasonal_multiplier = 1.3
            else:
                seasonal_multiplier = 1.0

            # No-sale probability (5%)
            if random.random() < 0.05:
                qty = 0
                price = 0
            else:
                qty = max(1, int(base_qty * weekend_multiplier *
                          seasonal_multiplier * random.uniform(0.8, 1.2)))
                # Price variation ±10%
                price = product['sell_price'] * random.uniform(0.9, 1.1)

            if qty > 0:
                expanded_transactions.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'store_id': int(store_id),
                    'sku_id': sku_id,
                    'qty_sold': int(qty),
                    'transaction_price': float(round(price, 2))
                })

    current_date += timedelta(days=1)

expanded_df = pd.DataFrame(expanded_transactions)
print(f"✅ Generated {len(expanded_df):,} transactions")

# Merge with product info
expanded_df = expanded_df.merge(
    products_df[['sku_id', 'product_name',
                 'category', 'cost_price', 'sell_price']],
    on='sku_id',
    how='left'
)

# Calculate metrics
expanded_df['revenue'] = expanded_df['qty_sold'] * \
    expanded_df['transaction_price']
expanded_df['margin'] = (expanded_df['transaction_price'] -
                         expanded_df['cost_price']) * expanded_df['qty_sold']

# Save expanded dataset
expanded_df.to_csv('expanded_historical_transactions.csv', index=False)
print(f"💾 Saved expanded dataset to: expanded_historical_transactions.csv")

print(f"\n📊 EXPANDED DATASET STATISTICS:")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(
    f"Date range: {expanded_df['date'].min()} to {expanded_df['date'].max()}")
print(f"Total transactions: {len(expanded_df):,}")
print(f"Stores: {expanded_df['store_id'].nunique()}")
print(f"SKUs: {expanded_df['sku_id'].nunique()}")
print(f"Avg qty per transaction: {expanded_df['qty_sold'].mean():.2f}")
print(f"Total revenue: ${expanded_df['revenue'].sum():,.2f}")
