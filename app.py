import streamlit as st
import pandas as pd
import pickle

# Load model
with open("churn_model.pkl", "rb") as f:
    payload = pickle.load(f)

lr     = payload["model"]
scaler = payload["scaler"]

# App title
st.title("Credit Card Customer Churn Predictor")
st.write("Enter customer details below and click **Predict Churn**.")
st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
    credit_limit = st.number_input("Credit Limit ($)", min_value=0, max_value=200000, value=10000, step=500)
    total_transactions = st.number_input("Total Transactions (last year)", min_value=0, max_value=200, value=40)

with col2:
    avg_utilization = st.slider("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    late_payments = st.number_input("Late Payments (last year)", min_value=0, max_value=20, value=1)
    tenure_years = st.number_input("Tenure (Years)", min_value=0, max_value=30, value=5)

st.markdown("---")

# Predict button
if st.button("Predict Churn"):

    # 1. Take inputs
    input_data = [[
        age,
        income,
        credit_limit,
        transactions,
        utilization,
        late_payments,
        tenure
    ]]

    # 2. Scale input
    input_scaled = scaler.transform(input_data)

    # 3. Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # 4. Output
    if prediction == 1:
        st.error(f"⚠️ Customer WILL churn  (Probability: {probability:.1%})")

        st.write("###  Business Recommendation:")
        st.write("""
        - Offer targeted retention incentives
        - Provide EMI / reduce financial burden
        - Personalized outreach
        """)

    else:
        st.success(f"✅ Customer will NOT churn  (Probability: {probability:.1%})")

        st.write("###  Business Recommendation:")
        st.write("""
        - Upsell premium products
        - Increase engagement
        - Cross-sell services
        """)
    st.write("**Input Summary:**")
    st.dataframe(input_data)

st.markdown("---")
