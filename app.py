import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================================
# LOAD MODEL
# ================================
with open("churn_model.pkl", "rb") as f:
    payload = pickle.load(f)

model  = payload["model"]
scaler = payload["scaler"]

# ================================
# RAG KNOWLEDGE BASE
# ================================
faq_data = [
    "High utilization ratio above 70% indicates financial stress and increases churn risk.",
    "Customers with more than 5 late payments are highly likely to churn.",
    "Low transaction activity means the customer is disengaged.",
    "Retention strategies include cashback, EMI options, and proactive outreach.",
    "Non-churn customers should be targeted for upselling and cross-selling.",
    "High tenure customers are usually more loyal.",
    "Low utilization and high transactions indicate a healthy customer."
]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = embed_model.encode(faq_data)

def retrieve_context(query):
    query_embedding = embed_model.encode([query])[0]
    scores = np.dot(faq_embeddings, query_embedding)
    best_idx = np.argmax(scores)
    return faq_data[best_idx]

# ================================
# APP UI
# ================================
st.title("💳 Credit Card Churn Prediction + AI Insights")
st.write("Enter customer details and get churn prediction + business insights.")
st.markdown("---")

# INPUTS
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    annual_income = st.number_input("Annual Income ($)", 0, 500000, 50000, step=1000)
    credit_limit = st.number_input("Credit Limit ($)", 0, 200000, 10000, step=500)
    total_transactions = st.number_input("Total Transactions", 0, 200, 40)

with col2:
    avg_utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.3)
    late_payments = st.number_input("Late Payments", 0, 20, 1)
    tenure_years = st.number_input("Tenure (Years)", 0, 30, 5)

st.markdown("---")

# ================================
# PREDICTION
# ================================
if st.button("Predict Churn"):

    input_data = [[
        age,
        annual_income,
        credit_limit,
        total_transactions,
        avg_utilization,
        late_payments,
        tenure_years
    ]]

    input_df = pd.DataFrame(input_data, columns=[
        "Age", "Annual_Income", "Credit_Limit",
        "Total_Transactions", "Avg_Utilization_Ratio",
        "Late_Payments", "Tenure_Years"
    ])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ================================
    # OUTPUT
    # ================================
    if prediction == 1:
        st.error(f"⚠️ Customer WILL churn (Probability: {probability:.1%})")

        if probability > 0.8:
            st.warning("🔥 High Risk Customer – Immediate Action Required")

        st.write("### 📌 Business Recommendation:")
        st.write("""
        - Offer targeted retention incentives (cashback, fee waiver)
        - Provide EMI options to reduce financial burden
        - Proactively reach out to customer
        - Monitor high utilization & late payments
        """)

    else:
        st.success(f"✅ Customer will NOT churn (Probability: {probability:.1%})")

        st.write("### 📌 Business Recommendation:")
        st.write("""
        - Upsell premium credit cards
        - Increase engagement via rewards
        - Cross-sell loans or insurance
        - Maintain relationship with personalized offers
        """)

    st.write("### 📊 Input Summary:")
    st.dataframe(input_df)

st.markdown("---")

# ================================
# GEN AI FAQ (RAG)
# ================================
st.markdown("## 🤖 AI FAQ Assistant")

user_query = st.text_input("Ask a question about customer behavior or churn:")

if st.button("Get AI Insight"):
    if user_query:
        context = retrieve_context(user_query)

        st.write("### 💡 Insight:")
        st.write(context)
    else:
        st.warning("Please enter a question.")
