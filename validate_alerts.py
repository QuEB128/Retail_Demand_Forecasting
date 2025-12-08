import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
import os


def load_artifacts(path='xgboost_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def generate_forecast(store_id, sku_id, features_df, artifacts, days=30):
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_cols = artifacts['feature_cols']

    store_sku = features_df[(features_df['store_id'] == store_id) & (
        features_df['sku_id'] == sku_id)].copy()
    store_sku = store_sku.sort_values('date')
    if len(store_sku) == 0:
        return None

    preds = []
    current_date = store_sku['date'].max() + pd.Timedelta(days=1)

    for i in range(days):
        forecast_date = current_date + pd.Timedelta(days=i)
        latest = store_sku.iloc[-1].copy()

        # update simple temporal features used by model
        latest['date'] = forecast_date
        latest['day_of_week'] = forecast_date.dayofweek
        latest['month'] = forecast_date.month
        latest['day_of_year'] = forecast_date.dayofyear
        latest['is_weekend'] = 1 if forecast_date.dayofweek >= 5 else 0
        latest['day_of_week_sin'] = np.sin(
            2 * np.pi * forecast_date.dayofweek / 7)
        latest['day_of_week_cos'] = np.cos(
            2 * np.pi * forecast_date.dayofweek / 7)
        latest['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
        latest['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)

        # Prepare X using feature_cols; if missing columns, fill with 0
        X_row = []
        for c in feature_cols:
            if c in latest.index:
                X_row.append(float(latest[c]))
            else:
                X_row.append(0.0)

        X = np.array(X_row).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = max(0.0, float(model.predict(X_scaled)[0]))
        preds.append(pred)

        # append latest so next day uses updated lags (approximation)
        store_sku = pd.concat(
            [store_sku, pd.DataFrame([latest])], ignore_index=True)

    return sum(preds), preds


def calculate_alert(store_id, sku_id, cleaned_df, features_df, artifacts, inventory_df, forecast_days=30):
    product = cleaned_df[cleaned_df['sku_id'] == sku_id]
    if len(product) == 0:
        return None

    demand_period, preds = generate_forecast(
        store_id, sku_id, features_df, artifacts, days=forecast_days)
    if demand_period is None:
        return None

    inv = inventory_df[(inventory_df['store_id'] == store_id)
                       & (inventory_df['sku_id'] == sku_id)]
    if len(inv) == 0:
        current_stock = 50
        lead_time = 7
    else:
        current_stock = int(inv['stock_on_hand'].values[0])
        lead_time = int(inv['lead_time_days'].values[0]
                        ) if 'lead_time_days' in inv.columns else 7

    daily_avg = demand_period / forecast_days if forecast_days > 0 else 1
    reorder_point = daily_avg * lead_time + daily_avg * 7
    stock_after_demand = current_stock - demand_period

    if stock_after_demand < 0:
        status = 'CRITICAL REORDER'
        overstock_value = 0
    elif current_stock < reorder_point:
        status = 'LOW STOCK WARNING'
        overstock_value = 0
    elif current_stock > demand_period * 2:
        excess_stock = current_stock - (demand_period * 1.2)
        overstock_value = max(0, excess_stock * 50)
        status = 'OVERSTOCK WATCH'
    elif current_stock >= demand_period * 0.8 and current_stock <= demand_period * 1.2:
        status = 'BALANCED - NORMAL'
        overstock_value = 0
    else:
        status = 'NORMAL'
        overstock_value = 0

    return {
        'store_id': store_id,
        'sku_id': sku_id,
        'current_stock': int(current_stock),
        'demand_period': round(demand_period, 2),
        'shortage': max(0, round(demand_period - current_stock, 2)),
        'status': status,
        'overstock_value': round(overstock_value, 2)
    }


if __name__ == '__main__':
    artifacts = load_artifacts('xgboost_model.pkl')
    cleaned_df = pd.read_csv('cleaned_data.csv')
    features_df = pd.read_csv('features_engineered_data.csv')
    features_df['date'] = pd.to_datetime(features_df['date'])
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    if os.path.exists('current_inventory.csv'):
        inventory_df = pd.read_csv('current_inventory.csv')
    else:
        inventory_df = pd.DataFrame()

    # Choose a few sample store/SKUs to demonstrate effect of forecast horizon
    sample_pairs = [(1, 8801), (2, 5502), (3, 2205)]
    horizons = [7, 30, 60]

    for store_id, sku_id in sample_pairs:
        print(f"\n== Store {store_id} - SKU {sku_id} ==")
        for h in horizons:
            alert = calculate_alert(
                store_id, sku_id, cleaned_df, features_df, artifacts, inventory_df, forecast_days=h)
            print(
                f"Horizon={h:2d} days --> demand={alert['demand_period']}, stock={alert['current_stock']}, shortage={alert['shortage']}, status={alert['status']}, overstock_value=${alert['overstock_value']}")
