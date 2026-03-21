# Credit Card Customer Churn Prediction

End-to-end ML project using **Logistic Regression** + **Streamlit**.

## Files

| File | Description |
|---|---|
| `churn_model.ipynb` | EDA + model training notebook |
| `app.py` | Streamlit web app |
| `churn_model.pkl` | Saved trained model |
| `churn_data.csv` | Synthetic dataset (500 rows) |
| `requirements.txt` | Python dependencies |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

- Algorithm: Logistic Regression
- Scaler: MinMaxScaler
- Test Accuracy: ~79%
- Dataset: 500 rows, 27% churn rate

URL
https://credit-card-churn-prediction-a7e7srnlptap32eemxzan9.streamlit.app/
