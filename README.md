# Retail Demand Forecasting Capstone Project

## Overview
This project implements a machine learning solution to forecast retail product demand using historical sales data. The system analyzes various factors including sales metrics, profit margins, temporal patterns, and product categories to predict order quantities with high accuracy.

## Project Objectives
- Build a predictive model to forecast product quantity demand
- Identify key factors driving demand decisions
- Evaluate model performance using industry-standard metrics
- Provide actionable insights for inventory and supply chain management

## Dataset
The project uses retail sales data from `stores_sales_forecasting.csv` containing:
- **9,994 records** of historical transactions
- **21 features** including:
  - Transaction details (Order Date, Ship Date, Customer info)
  - Product information (Category, Sub-Category, Product name)
  - Financial metrics (Sales, Discount, Profit)
  - Geographic data (Region, State, City, Country)
  - Logistics (Ship Mode)

## Features & Preprocessing

### Date Features
- `day`: Day of month (1-31)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `month`: Month (1-12)
- `year`: Year of transaction

### Categorical Encoding
One-hot encoding applied to 8 categorical variables:
- Ship Mode, Segment, Country, City, State, Region, Category, Sub-Category
- Resulting in 436 total features after encoding

### Feature Engineering
- Extracted temporal components from Order Date and Ship Date
- Converted categorical variables into numerical representations
- Dropped non-predictive columns (IDs, names, dates)

## Model & Results

### Algorithm: Random Forest Regressor
**Configuration:**
- Number of estimators: 300 trees
- Max depth: 20 levels
- Random state: 42 (for reproducibility)

### Performance Metrics
- **MAE (Mean Absolute Error)**: 1.46 units
  - Model predictions deviate by ~1.46 units on average
- **RMSE (Root Mean Squared Error)**: 2.00 units
  - Penalizes larger errors, showing good overall prediction quality

### Top 20 Most Important Features
1. **Sales** (34.8%) - Primary predictor of order quantity
2. **Profit** (11.1%) - Profitability strongly influences demand
3. **Sub-Category_Furnishings** (6.0%) - Product type matters
4. **Postal Code** (5.0%) - Geographic location factor
5. **month** (4.5%) - Seasonal patterns present
6. **day_of_week** (3.5%) - Weekly patterns influence orders
7. And 14 more features with decreasing importance

## File Structure
```
Retail_Demand_Forecasting/
├── retail_demand_forecasting.ipynb  # Main Jupyter notebook with full pipeline
├── stores_sales_forecasting.csv     # Training dataset (9,994 records)
├── mini_mart_dataset_500.csv        # Alternative smaller dataset
└── README.md                        # Project documentation
```

## Project Workflow

### 1. Data Loading & Exploration
```python
df = pd.read_csv("stores_sales_forecasting.csv", encoding="latin1")
```

### 2. Feature Engineering
- Parse date columns and extract temporal features
- Apply one-hot encoding to categorical variables
- Remove non-predictive columns

### 3. Train-Test Split
- 80% training data, 20% test data
- Random state 42 for reproducibility

### 4. Model Training
- Initialize RandomForestRegressor with 300 estimators
- Train on processed features

### 5. Evaluation
- Calculate MAE and RMSE on test set
- Analyze feature importance scores
- Generate predictions on new samples

## Key Insights

1. **Sales is Dominant**: Sales revenue is the single strongest predictor, accounting for nearly 35% of model importance
2. **Profitability Matters**: Second most important feature (11%), indicating profit-driven purchasing decisions
3. **Geographic & Temporal Patterns**: Month, day of week, and postal code show seasonal and regional demand variations
4. **Product Segmentation**: Furnishings and other product categories have distinct demand patterns
5. **Prediction Accuracy**: Low MAE and RMSE indicate the model is suitable for practical demand forecasting

## Usage

### Running the Notebook
1. Ensure all dependencies are installed
2. Open `retail_demand_forecasting.ipynb` in Jupyter
3. Execute cells sequentially to:
   - Load and preprocess data
   - Train the model
   - Generate predictions and visualizations

### Making Predictions
```python
# Get model prediction for a sample
sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]
print(f"Predicted Quantity: {prediction:.2f} units")
```

## Dependencies
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning (RandomForestRegressor, train_test_split, metrics)
- **matplotlib** - Data visualization

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Performance Implications

### Inventory Management
- Forecast accuracy enables better inventory levels
- Reduces stockouts and overstocking costs
- Typical prediction error of ±1.46 units supports 4-5% demand accuracy

### Business Value
- Supports data-driven purchasing decisions
- Identifies seasonal demand patterns
- Enables regional demand customization
- Improves supply chain efficiency

## Future Enhancements
- Time series forecasting (ARIMA, Prophet) for temporal patterns
- Gradient Boosting alternatives (XGBoost, LightGBM)
- Feature importance analysis for actionable business insights
- Hyperparameter tuning with cross-validation
- Ensemble methods combining multiple models
- Segment-specific models for different product categories

## Author
Retail Analytics Team

## Date
November 2025
