import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import config
import json
import datetime
import numpy as np

from modules.weather_service import fetch_live_weather
from modules.preprocessing import split_and_scale
from modules.data_loader import validate_data

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Solar Power Monitoring System",
    layout="wide"
)

# ==============================================================
# CUSTOM UI
# ==============================================================

st.markdown("""
<style>
.big-title {font-size:38px;font-weight:bold;color:#FDB813;}
.subtitle {font-size:18px;color:#AAAAAA;}
.prediction-box {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    padding:30px;border-radius:15px;
    text-align:center;font-size:32px;
    font-weight:bold;color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Solar Power Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ultra-Short-Term (15 Min) Solar Power Prediction Dashboard</div>', unsafe_allow_html=True)
st.divider()

# ==============================================================
# LOAD MODEL
# ==============================================================

@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)

model = load_model()

# Load feature columns
feat_path = getattr(config, "FEATURE_COLUMNS_PATH", "models/feature_columns.json")
try:
    with open(feat_path, "r") as fh:
        feature_cols = json.load(fh)
except:
    feature_cols = ["Temperature", "Humidity", "Cloud_Cover", "Wind_Speed", "Hour", "DayOfWeek", "Month"]

# ==============================================================
# SIDEBAR
# ==============================================================

LIVE_MODE = "Live Weather Prediction"
UPLOAD_MODE = "Upload Dataset Prediction"
COMPARE_MODE = "Model Comparison & Analysis"
ACTUAL_MODE = "Actual vs Predicted Analysis"
FORECAST_MODE = "Next 15_min forecast"

st.sidebar.title("⚙ Control Panel")

mode = st.sidebar.radio(
    "Select Mode",
    [
        LIVE_MODE,
        UPLOAD_MODE,
        COMPARE_MODE,
        ACTUAL_MODE,
        FORECAST_MODE
    ],
)

# ==============================================================
# 1️⃣ LIVE WEATHER MODE
# ==============================================================

if mode == LIVE_MODE:

    st.subheader("🔮 Live Weather Prediction")

    if st.button("🚀 Fetch Weather & Predict"):

        weather = fetch_live_weather() or {}
        now = datetime.datetime.now()

        row = {
            "Temperature": weather.get("Temperature", 0),
            "Humidity": weather.get("Humidity", 0),
            "Cloud_Cover": weather.get("Cloud_Cover", 0),
            "Wind_Speed": weather.get("Wind_Speed", 0),
            "Hour": now.hour,
            "DayOfWeek": now.weekday(),
            "Month": now.month
        }

        for col in feature_cols:
            if col.startswith("Power_lag_"):
                row[col] = 0.0

        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        prediction = model.predict(input_df)[0]

        # Show metrics AFTER prediction
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡 Temp (°C)", round(row["Temperature"],2))
        col2.metric("💧 Humidity (%)", round(row["Humidity"],2))
        col3.metric("☁ Cloud Cover (%)", round(row["Cloud_Cover"],2))
        col4.metric("🌬 Wind Speed (m/s)", round(row["Wind_Speed"],2))

        st.divider()

        st.markdown(
            f'<div class="prediction-box">⚡ {round(prediction,2)} kW</div>',
            unsafe_allow_html=True
        )

# ==============================================================
# 2️⃣ DATASET UPLOAD MODE
# ==============================================================

elif mode == UPLOAD_MODE:

    st.subheader("📂 Upload Dataset for Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        df = df.copy()

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        df_model = df[feature_cols]
        df_model = df_model.reindex(columns=feature_cols, fill_value=0)

        predictions = model.predict(df_model)

        df["Predicted_Power_kW"] = predictions

        st.write("### Prediction Results")
        st.dataframe(df.head(), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Results", csv, "predicted_results.csv")

        fig = px.line(df, y="Predicted_Power_kW", title="Predicted Solar Power Trend")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 3️⃣ MODEL COMPARISON
# ==============================================================

elif mode == COMPARE_MODE:

    st.subheader("📊 Model Performance Comparison")

    try:
        performance = pd.read_csv(config.PERFORMANCE_PATH)
        st.dataframe(performance, use_container_width=True)

        fig = px.bar(performance, x="Model", y="RMSE", text="RMSE")
        st.plotly_chart(fig, use_container_width=True)

        best = performance.loc[performance["RMSE"].idxmin()]
        st.success(f"🏆 Best Model: {best['Model']} | RMSE: {best['RMSE']}")

    except:
        st.warning("Run train.py first to generate performance file.")

# ==============================================================
# 4️⃣ ACTUAL VS PREDICTED
# ==============================================================

elif mode == ACTUAL_MODE:

    st.subheader("📈 Actual vs Predicted Analysis")

    try:
        df = pd.read_csv("solar_dataset_multi_year_3years.csv")
        df = validate_data(df)

        X_train, X_test, y_train, y_test = split_and_scale(df)
        y_pred = model.predict(X_test)

        comparison_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        }).reset_index(drop=True)

        st.dataframe(comparison_df.head(50), use_container_width=True)

        fig_line = px.line(comparison_df.head(200), y=["Actual","Predicted"])
        st.plotly_chart(fig_line, use_container_width=True)

        fig_scatter = px.scatter(comparison_df, x="Actual", y="Predicted")
        st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ==============================================================
# 5️⃣ NEXT 15-MIN FORECAST
# ==============================================================

elif mode == FORECAST_MODE:

    st.subheader("⏳ Next 15-Minute Solar Power Forecast")

    if st.button("🔮 Generate Forecast"):

        now = datetime.datetime.now()
        forecast_data = []

        weather = fetch_live_weather() or {}

        base_temp = weather.get("Temperature", 30)
        base_humidity = weather.get("Humidity", 60)
        base_cloud = weather.get("Cloud_Cover", 20)
        base_wind = weather.get("Wind_Speed", 5)

        for step in range(1, 16):

            future_time = now + datetime.timedelta(minutes=step)

            row = {
                 "Temperature": base_temp,
                 "Humidity": base_humidity,
                 "Cloud_Cover": base_cloud,
                 "Wind_Speed": base_wind,
                 "Hour": future_time.hour,
                 "DayOfWeek": future_time.weekday(),
                 "Month": future_time.month
             }
 
            for col in feature_cols:
                 if col.startswith("Power_lag_"):
                     row[col] = 0.0
 
            input_df = pd.DataFrame([row])
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)
            prediction = model.predict(input_df)[0]
 
            forecast_data.append({
                 "Time": future_time.strftime("%H:%M"),
                 "Forecasted Power (kW)": prediction
             })
 
        forecast_df = pd.DataFrame(forecast_data)
 
        st.dataframe(forecast_df, use_container_width=True)
 
        fig = px.line(
             forecast_df,
             x="Time",
             y="Forecasted Power (kW)",
             markers=True,
             title="Next 15-Minute Solar Forecast"
    ) 
    st.plotly_chart(fig, use_container_width=True)



