import pandas as pd

hist = pd.read_csv('historical_transactions.csv')
products = pd.read_csv('product_hierarchy.csv')

print('hist columns:', hist.columns.tolist())
merged = hist.merge(products[['sku_id', 'product_name', 'category',
                    'cost_price', 'sell_price']], on='sku_id', how='left')
print('merged columns:', merged.columns.tolist())
print('cost_price present in merged?', 'cost_price' in merged.columns)
print('sample cost_price values (unique, dropna):',
      merged['cost_price'].dropna().unique()[:5])
print('any nulls in cost_price?', merged['cost_price'].isnull().any())
print('dtype of sku_id in hist:', hist['sku_id'].dtype)
print('dtype of sku_id in products:', products['sku_id'].dtype)
