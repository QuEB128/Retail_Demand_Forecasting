"""
Step 3: XGBoost Model Training
Trains demand forecasting model with time-based train/test split
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 3: XGBOOST MODEL TRAINING")
print("=" * 60)

# Load engineered features
print("\n📥 Loading engineered features...")
df = pd.read_csv('features_engineered_data.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"   Loaded {len(df):,} records")

# Select features for model
print("\n🎯 Selecting features for model...")
exclude_cols = ['date', 'product_name', 'category', 'qty_sold', 'revenue', 'transaction_price']
X_cols = [col for col in df.columns if col not in exclude_cols]
y_col = 'qty_sold'

X = df[X_cols].astype(float)
y = df[y_col].astype(float)

print(f"   Features: {len(X_cols)}")
print(f"   Target variable: {y_col}")
print(f"   Data points: {len(X):,}")

# Time-based train/test split (80/20)
print("\n⏱️  Time-based train/test split...")
split_idx = int(len(df) * 0.8)
train_cutoff = df.iloc[split_idx]['date']

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]
df_test = df.iloc[split_idx:].copy()

print(f"   Training set: {len(X_train):,} records (up to {train_cutoff.date()})")
print(f"   Test set: {len(X_test):,} records (from {df_test['date'].min().date()} to {df_test['date'].max().date()})")
print(f"   Train/Test split: {len(X_train)/len(X)*100:.1f}% / {len(X_test)/len(X)*100:.1f}%")

# Feature scaling
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   ✓ Features standardized")

# XGBoost model configuration
print("\n⚙️  Configuring XGBoost model...")
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
print(f"   ✓ Hyperparameters set")
for param, value in xgb_params.items():
    if param != 'n_jobs':
        print(f"      {param}: {value}")

# Train model
print("\n🚀 Training XGBoost model...")
model = xgb.XGBRegressor(**xgb_params, n_estimators=200, early_stopping_rounds=20, verbose=0)
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)
print(f"   ✓ Model trained with {model.n_estimators} estimators")

# Predictions
print("\n🔮 Generating predictions...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
print(f"   ✓ Predictions generated")

# Model evaluation
print("\n📊 Model Performance Evaluation:")
print("\n   TRAINING SET:")
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

print(f"      R² Score: {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"      RMSE: {train_rmse:.4f} units")
print(f"      MAE: {train_mae:.4f} units")
print(f"      MAPE: {train_mape:.2f}%")

print("\n   TEST SET:")
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

print(f"      R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"      RMSE: {test_rmse:.4f} units")
print(f"      MAE: {test_mae:.4f} units")
print(f"      MAPE: {test_mape:.2f}%")

# Feature importance
print("\n🎯 Feature Importance (Top 15):")
feature_importance = pd.DataFrame({
    'feature': X_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"   {idx:2d}. {row['feature']:25s} {row['importance']:.4f}")

# Performance by store
print("\n🏪 Test Performance by Store:")
df_test['pred'] = y_test_pred
for store_id in sorted(df_test['store_id'].unique()):
    store_data = df_test[df_test['store_id'] == store_id]
    r2 = r2_score(store_data['qty_sold'], store_data['pred'])
    rmse = np.sqrt(mean_squared_error(store_data['qty_sold'], store_data['pred']))
    print(f"   Store {store_id}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

# Performance by SKU
print("\n📦 Test Performance by SKU (Top 5):")
sku_performance = []
for sku_id in df_test['sku_id'].unique():
    sku_data = df_test[df_test['sku_id'] == sku_id]
    if len(sku_data) > 20:
        r2 = r2_score(sku_data['qty_sold'], sku_data['pred'])
        sku_performance.append({'sku_id': sku_id, 'r2': r2, 'count': len(sku_data)})

sku_perf_df = pd.DataFrame(sku_performance).sort_values('r2', ascending=False)
for _, row in sku_perf_df.head(5).iterrows():
    print(f"   SKU {int(row['sku_id'])}: R² = {row['r2']:.4f} ({int(row['count'])} test records)")

# Residual analysis
print("\n📈 Residual Analysis:")
residuals = y_test - y_test_pred
print(f"   Mean residual: {residuals.mean():.4f}")
print(f"   Std dev residual: {residuals.std():.4f}")
print(f"   Min residual: {residuals.min():.4f}")
print(f"   Max residual: {residuals.max():.4f}")
print(f"   Residuals within ±1 RMSE: {(np.abs(residuals) <= test_rmse).sum() / len(residuals) * 100:.1f}%")

# Save model
print("\n💾 Saving model...")
model_file = 'xgboost_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': X_cols,
        'training_date': datetime.now().isoformat(),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse
    }, f)
print(f"   ✓ Model saved to: {model_file}")

# Save feature importance
importance_file = 'feature_importance.csv'
feature_importance.to_csv(importance_file, index=False)
print(f"   ✓ Feature importance saved to: {importance_file}")

# Save test predictions for analysis
test_results = df_test[['date', 'store_id', 'sku_id', 'qty_sold', 'product_name']].copy()
test_results['predicted_qty'] = y_test_pred
test_results['residual'] = y_test - y_test_pred
test_results['abs_error_pct'] = np.abs((y_test - y_test_pred) / (y_test + 1)) * 100
test_results = test_results.sort_values('date')
test_results.to_csv('model_test_results.csv', index=False)
print(f"   ✓ Test results saved to: model_test_results.csv")

print("\n✅ Model Summary:")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Features used: {len(X_cols)}")
print(f"   Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"   Test RMSE: {test_rmse:.4f} units")
print(f"   Model file: {model_file}")

print("\n" + "=" * 60)
print("✅ STEP 3 COMPLETE")
print("=" * 60)
