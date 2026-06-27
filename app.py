import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

st.title("📈 Sales Forecasting System")
st.markdown("**Sqrock IT Solutions Internship — Project 1** | Built with Random Forest Regressor")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your Superstore CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Month'] = df['Order Date'].dt.month
    df['Year']  = df['Order Date'].dt.year
    df['Day']   = df['Order Date'].dt.day

    st.success(f"Dataset loaded! {df.shape[0]} rows, {df.shape[1]} columns")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales",    f"${df['Sales'].sum():,.0f}")
    col2.metric("Average Order",  f"${df['Sales'].mean():,.2f}")
    col3.metric("Total Orders",   f"{df.shape[0]:,}")

    st.subheader("📊 Monthly Sales Trend")
    monthly = df.groupby(['Year','Month'])['Sales'].sum().reset_index()
    monthly['Date'] = pd.to_datetime(monthly[['Year','Month']].assign(Day=1))
    fig1, ax1 = plt.subplots(figsize=(12,4))
    ax1.plot(monthly['Date'], monthly['Sales'], marker='o', color='steelblue')
    ax1.set_title('Monthly Sales Trend')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("🏷️ Sales by Category")
    fig2, ax2 = plt.subplots(figsize=(7,3))
    df.groupby('Category')['Sales'].sum().sort_values().plot(kind='barh', color='coral', ax=ax2)
    ax2.set_xlabel('Sales ($)')
    st.pyplot(fig2)

    st.subheader("🤖 Model Training & Prediction")
    X = df[['Month','Year','Day']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    c1, c2 = st.columns(2)
    c1.metric("MAE  (Mean Absolute Error)",  f"${mae:.2f}")
    c2.metric("RMSE (Root Mean Sq Error)",   f"${rmse:.2f}")

    st.subheader("📉 Actual vs Predicted Sales")
    fig3, ax3 = plt.subplots(figsize=(12,4))
    ax3.plot(y_test.values[:50], label='Actual',    color='steelblue', marker='o')
    ax3.plot(preds[:50],         label='Predicted', color='orange',    marker='x')
    ax3.set_title('Actual vs Predicted Sales (first 50 samples)')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Sales ($)')
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("🔮 Predict Future Sales")
    st.markdown("Enter a future date to predict sales:")
    col_m, col_y, col_d = st.columns(3)
    month = col_m.selectbox("Month", list(range(1,13)))
    year  = col_y.selectbox("Year",  [2024, 2025, 2026, 2027])
    day   = col_d.selectbox("Day",   list(range(1,32)))

    if st.button("Predict Sales 🚀"):
        prediction = model.predict([[month, year, day]])[0]
        st.success(f"Predicted Sales for {day}/{month}/{year}: **${prediction:,.2f}**")

else:
    st.info("👆 Please upload the 'Sample - Superstore.csv' file to get started.")
    st.markdown("""
    **How to use:**
    1. Download the Superstore dataset from Kaggle
    2. Upload the CSV file above
    3. The app will auto-train the model and show predictions
    """)
