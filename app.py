import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("fraud_detection_model.json")


ohe = joblib.load('onehot_encoder.pkl')
feature_order = joblib.load('feature_columns.pkl')

# Kolom input
cat_cols = ['Account_Type', 'Transaction_Type', 'Merchant_Category', 'Device_Type']
num_cols = ['Age', 'Transaction_Amount', 'Account_Balance', 'Gender']

def main():
    st.title('Fraud Detection App')

    # Input dari user
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    account_balance = st.number_input("Account Balance", min_value=0.0)
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
    account_type = st.selectbox("Account Type", ["Savings", "Checking", "Business", "Loan"])
    transaction_type = st.selectbox("Transaction Type", ["Transfer", "Bill Payment", "Withdrawal", "Debit", "Credit"])
    merchant_category = st.selectbox("Merchant Category", ["Restaurant", "Groceries", "Entertainment", "Clothing", "Health", "Electronics"])
    device_type = st.selectbox("Device Type", ["POS", "ATM", "Desktop", "Mobile"])

    if st.button("Predict Fraud"):
        input_data = {
            'Age': age,
            'Transaction_Amount': transaction_amount,
            'Account_Balance': account_balance,
            'Gender': 1 if gender == "Male" else 0,
            'Account_Type': account_type,
            'Transaction_Type': transaction_type,
            'Merchant_Category': merchant_category,
            'Device_Type': device_type
        }

        df = pd.DataFrame([input_data])

        # One-hot encoding
        encoded_cats = pd.DataFrame(
            ohe.transform(df[cat_cols]),
            columns=ohe.get_feature_names_out(cat_cols),
            index=df.index
        )

        # Gabungkan data numerik dan hasil encoding
        df_final = pd.concat([df[num_cols], encoded_cats], axis=1)

        # Tambahkan kolom yang mungkin hilang agar sesuai urutan
        for col in feature_order:
            if col not in df_final.columns:
                df_final[col] = 0
        df_final = df_final[feature_order]

        # Prediksi
        proba = model.predict_proba(df_final)[0][1]
        prediction = model.predict(df_final)[0]

        if prediction == 1:
            st.error(f"⚠️ Fraud Detected! Probability: {proba:.2f}")
        else:
            st.success(f"✅ Legitimate Transaction. Probability of fraud: {proba:.2f}")

if __name__ == '__main__':
    main()
