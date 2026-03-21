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
if st.button("Predict Churn", use_container_width=True):

    input_data = pd.DataFrame([{
        "Age"                   : age,
        "Annual_Income"         : annual_income,
        "Credit_Limit"          : credit_limit,
        "Total_Transactions"    : total_transactions,
        "Avg_Utilization_Ratio" : avg_utilization,
        "Late_Payments"         : late_payments,
        "Tenure_Years"          : tenure_years
    }])

    input_scaled = scaler.transform(input_data)
    prediction   = lr.predict(input_scaled)[0]
    probability  = lr.predict_proba(input_scaled)[0][1]

    st.markdown("---")
if prediction == 1:
    st.error(f"⚠️ Customer WILL churn  (Probability: {probability:.1%})")
    
    st.write("### 📌 Business Recommendation:")
    st.write("""
    - Offer targeted retention incentives (cashback, fee waiver, reward points)
    - Reduce financial stress by increasing credit limit or offering EMI options
    - Proactively reach out via relationship manager or call center
    - Monitor high-risk behavior (high utilization, late payments)
    - Provide personalized offers based on spending pattern
    """)

else:
    st.success(f"✅ Customer will NOT churn  (Probability: {probability:.1%})")
    
    st.write("### 📌 Business Recommendation:")
    st.write("""
    - Continue engagement through loyalty programs
    - Upsell premium credit cards or higher credit limits
    - Encourage more transactions via offers and rewards
    - Maintain strong customer relationship through regular communication
    - Cross-sell products like loans, insurance, or investments
    """)
    
    st.write("**Input Summary:**")
    st.dataframe(input_data)

st.markdown("---")
