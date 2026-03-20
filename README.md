# Credit Card Customer Churn Prediction
### End-to-End ML Project | Logistic Regression + Streamlit

---

## Project Overview

Predict whether a credit card customer will **churn** based on financial behaviour and history.

---

## Files in this Repository

| File | Description |
|---|---|
| `churn_model.ipynb` | Full notebook — EDA + training + evaluation |
| `app.py` | Streamlit prediction app |
| `churn_model.pkl` | Saved trained model |
| `churn_data.csv` | Synthetic dataset (500 rows) |
| `requirements.txt` | Python dependencies |

---

## Dataset Features

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `Annual_Income` | Yearly income ($) |
| `Credit_Limit` | Credit card limit ($) |
| `Total_Transactions` | Transactions last year |
| `Avg_Utilization_Ratio` | Credit used / available (0–1) |
| `Late_Payments` | Late payments last year |
| `Tenure_Years` | Years as card holder |
| `Churn` | **Target** — 0 = No, 1 = Yes |

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Open the notebook
```bash
jupyter notebook churn_model.ipynb
```

### Step 3 — Launch the app
```bash
streamlit run app.py
```

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Scaler | MinMaxScaler |
| Split | 80 / 20 |
| Test Accuracy | ~79% |

---

## Tech Stack
`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `streamlit` · `pickle`
