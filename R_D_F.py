import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from io import BytesIO

# ------------------------------------------
# Load dataset
# ------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    
    # Detect date column
    date_col = None
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[:5])
            date_col = c
            break
        except:
            continue

    # Detect product column
    product_candidates = ["product", "product_id", "item", "item_name", "sku"]
    product_col = None
    for c in df.columns:
        if c.lower() in product_candidates:
            product_col = c
            break

    # Detect sales column
    sales_candidates = ["units_sold", "sales", "quantity", "qty"]
    sales_col = None
    for c in df.columns:
        if c.lower() in sales_candidates:
            sales_col = c
            break

    # Fallback if missing
    if sales_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        sales_col = numeric_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    return df, date_col, product_col, sales_col


# ------------------------------------------
# Forecasting function
# ------------------------------------------
def forecast_model(df, date_col, product_col, sales_col, product, horizon):

    dff = df[df[product_col] == product].copy()
    if dff.empty:
        st.error("No records found for this product.")
        return None

    # Aggregate daily
    series = (
        dff.groupby(pd.Grouper(key=date_col, freq="D"))[sales_col]
        .sum()
        .asfreq("D", fill_value=0)
    )

    # Train-test split for evaluation
    test_size = min(14, max(1, int(len(series)*0.2)))
    train = series[:-test_size]
    test = series[-test_size:]

    # ExponentialSmoothing (Holt-Winters)
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated"
    )

    fit = model.fit()
    #forecast = fit.forecast(horizon)

    # Forecast values
    forecast_values = fit.forecast(horizon)

    # Force forecast to start from TODAY (not dataset end)
    start_date = pd.Timestamp.today().normalize()
    forecast_index = pd.date_range(start=start_date, periods=horizon, freq="D")

    forecast = pd.Series(forecast_values.values, index=forecast_index)


    # Compute MAE
    test_pred = fit.forecast(test_size)
    mae = np.mean(np.abs(test - test_pred))

    return series, forecast, mae


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.title("📈 Retail Demand Forecasting App")
st.write("Select a product and forecast horizon to generate a demand forecast.")

file = st.file_uploader("Upload Retail Dataset (.csv)", type=["csv"])

if file:
    df, date_col, product_col, sales_col = load_data(file)

    st.success("Dataset loaded successfully!")
    st.write(f"**Date column:** {date_col}")
    st.write(f"**Product column:** {product_col}")
    st.write(f"**Sales column:** {sales_col}")

    products = sorted(df[product_col].unique())
    selected_product = st.selectbox("Select Product", products)

    horizon = st.slider("Forecast Horizon (Days)", 7, 120, 30)

    if st.button("Generate Forecast"):
        with st.spinner("Training model & generating forecast..."):

            series, forecast, mae = forecast_model(
                df, date_col, product_col, sales_col,
                selected_product, horizon
            )

            # Plot chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(series.index, series.values, label="History")
            ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
            ax.legend()
            ax.set_title(f"Forecast for {selected_product}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")

            st.pyplot(fig)

            st.success(f"Forecast complete! MAE: {mae:.2f}")

            # Prepare downloadable forecast CSV
            
            #forecast_df = pd.DataFrame({
            #    "date": forecast.index,
            #    "forecast": forecast.values
            #})
            

            # Prepare downloadable forecast CSV
            forecast_df = pd.DataFrame({
                "date": forecast.index,
                "forecast": forecast.values
            })

            # --- NEW: Detailed forecast statistics ---
            min_units = forecast_df["forecast"].min()
            max_units = forecast_df["forecast"].max()
            avg_units = forecast_df["forecast"].mean()

            st.subheader("📊 Forecast Summary Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Minimum Units Expected", f"{min_units:.2f}")

            with col2:
                st.metric("Average Units Expected", f"{avg_units:.2f}")

            with col3:
                st.metric("Maximum Units Expected", f"{max_units:.2f}")

            # Download button
            csv = forecast_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{selected_product}.csv",
                mime="text/csv",
                key=f"download_{selected_product}"
            )



            

else:
    st.info("Upload a CSV file to begin.")
