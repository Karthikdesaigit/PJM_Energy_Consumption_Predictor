
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# Title
st.title("🔌 PJM Energy Consumption Forecasting App")

# Load the original hourly PJM dataset
@st.cache_data
def load_data():
    df = pd.read_csv(
        "PJMW_hourly.csv",  # uploaded filename
        parse_dates=["Datetime"],
        index_col="Datetime"
    )
    return df

df = load_data()

# Resample to daily average
daily_df = df['PJMW_hourly'].resample('D').mean().to_frame()
daily_df.columns = ['y']  # Rename for consistency
daily_df.index.name = 'ds'

# Feature Engineering
def create_features(df):
    df = df.copy()
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    
    # Lag features
    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['y'].shift(lag)
        
    df.dropna(inplace=True)
    return df

df_feat = create_features(daily_df)
X = df_feat.drop('y', axis=1)
y = df_feat['y']

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# User Input: Future Days to Forecast
n_days = st.slider("Select number of future days to forecast:", 1, 30, 7)

# Create future dates
future_dates = pd.date_range(start=df_feat.index[-1] + timedelta(days=1), periods=n_days)
future_df = pd.DataFrame(index=future_dates)

# Start from last 7 days
recent = df_feat.copy().iloc[-7:].copy()
predictions = []

for date in future_df.index:
    row = {
        'day': date.day,
        'dayofweek': date.dayofweek,
        'month': date.month,
        'year': date.year,
        'is_weekend': int(date.dayofweek in [5, 6]),
    }

    for lag in range(1, 8):
        row[f'lag_{lag}'] = recent['y'].iloc[-lag]

    input_df = pd.DataFrame([row])
    y_pred = model.predict(input_df)[0]
    predictions.append(y_pred)

    # Append to recent for recursive forecasting
    new_row = row.copy()
    new_row['y'] = y_pred
    recent = pd.concat([recent, pd.DataFrame([new_row], index=[date])])

# Plot
st.subheader("📈 Forecast Plot")
plt.figure(figsize=(12, 6))
plt.plot(daily_df.index[-60:], daily_df['y'].iloc[-60:], label='Historical')
plt.plot(future_dates, predictions, label='Forecast', linestyle='--', marker='o')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MW)')
plt.title('PJM Daily Energy Forecast')
plt.grid(True)
plt.legend()
st.pyplot(plt)

# Table
st.subheader("🔢 Forecasted Values")
forecast_table = pd.DataFrame({'Date': future_dates, 'Forecast (MW)': predictions})
st.dataframe(forecast_table)
