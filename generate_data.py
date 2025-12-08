import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Settings
NUM_STORES = 3
NUM_PRODUCTS = 5
START_DATE = '2023-01-01'
END_DATE = '2023-12-31'

# 1. Create Product Hierarchy (Metadata)
products = [
    {'sku_id': 5501, 'product_name': '4K LED Smart TV', 'category': 'Electronics', 'cost_price': 300, 'sell_price': 450},
    {'sku_id': 5502, 'product_name': 'Bluetooth Speaker', 'category': 'Electronics', 'cost_price': 25, 'sell_price': 60},
    {'sku_id': 8801, 'product_name': 'Running Shoes', 'category': 'Apparel', 'cost_price': 40, 'sell_price': 95},
    {'sku_id': 8802, 'product_name': 'Cotton T-Shirt', 'category': 'Apparel', 'cost_price': 8, 'sell_price': 20},
    {'sku_id': 2205, 'product_name': 'Coffee Maker', 'category': 'Home', 'cost_price': 45, 'sell_price': 85}
]
df_products = pd.DataFrame(products)
df_products.to_csv('product_hierarchy.csv', index=False)
print("✅ Created product_hierarchy.csv")

# 2. Create Historical Transactions (The Training Data)
transactions = []
current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

print("⏳ Generating transactions... (this may take a moment)")

while current_date <= end_date:
    for store in range(1, NUM_STORES + 1):
        for product in products:
            # --- Logic to make data look realistic ---
            
            # Base demand varies by product popularity (cheaper items sell more)
            base_demand = 50 if product['category'] == 'Apparel' else 10 
            
            # Weekend Multiplier (Sales go up on Sat/Sun)
            is_weekend = current_date.weekday() >= 5
            weekend_factor = 1.4 if is_weekend else 1.0
            
            # Seasonality (e.g., Sales spike in December)
            is_december = current_date.month == 12
            season_factor = 1.5 if is_december else 1.0
            
            # Random noise (some days are just slow or busy randomly)
            noise = np.random.normal(1.0, 0.2) # Mean 1.0, Std Dev 0.2
            
            # Calculate final quantity
            final_qty = int(base_demand * weekend_factor * season_factor * noise)
            
            # Ensure we don't have negative sales
            final_qty = max(0, final_qty)
            
            # Simulate "No Sale" days occasionally (5% chance)
            if random.random() < 0.05:
                final_qty = 0

            # Append to list
            if final_qty > 0: # Only record if a sale happened
                transactions.append([
                    current_date.strftime("%Y-%m-%d"),
                    store,
                    product['sku_id'],
                    final_qty,
                    product['sell_price']
                ])
    
    current_date += timedelta(days=1)

df_transactions = pd.DataFrame(transactions, columns=['date', 'store_id', 'sku_id', 'qty_sold', 'transaction_price'])
df_transactions.to_csv('historical_transactions.csv', index=False)
print(f"✅ Created historical_transactions.csv with {len(df_transactions)} rows")

# 3. Create Current Inventory (The Live Data)
inventory = []
for store in range(1, NUM_STORES + 1):
    for product in products:
        # Random stock level between 0 and 100
        current_stock = random.randint(0, 100)
        # Random lead time (how long to restock)
        lead_time = random.choice([3, 7, 14])
        
        inventory.append([store, product['sku_id'], current_stock, lead_time])

df_inventory = pd.DataFrame(inventory, columns=['store_id', 'sku_id', 'stock_on_hand', 'lead_time_days'])
df_inventory.to_csv('current_inventory.csv', index=False)
print("✅ Created current_inventory.csv")

print("\n🎉 All dummy datasets generated successfully!")
