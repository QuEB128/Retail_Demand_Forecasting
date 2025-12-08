
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

try:
    cleaned_df = pd.read_csv('cleaned_data.csv')
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    
    features_df = pd.read_csv('features_engineered_data.csv')
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    inventory_df = pd.read_csv('current_inventory.csv')
    
    print('✅ Data loaded successfully!')
    print(f'\nCleaned Data: {cleaned_df.shape}')
    print(f'Features Data: {features_df.shape}')
    print(f'Inventory Data: {inventory_df.shape}')
except Exception as e:
    print(f'❌ Error loading data: {e}')

print('\n📊 Cleaned Data Sample:')
print(cleaned_df.head())

print('\n📊 Inventory Data:')
print(inventory_df)
