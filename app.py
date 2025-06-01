import streamlit as st
import pandas as pd
import joblib
<<<<<<< HEAD
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("fraud_detection_model.json")


ohe = joblib.load('onehot_encoder.pkl')
feature_order = joblib.load('feature_columns.pkl')

# Kolom input
=======

# Load model and preprocessors
model = joblib.load('xgb_fraud_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # For target decoding
ohe = joblib.load('onehot_encoder.pkl')  # For categorical features

# Define columns
>>>>>>> 62a15c05933bc5872474dde66ca9553d569234f5
cat_cols = ['Account_Type', 'Transaction_Type', 'Merchant_Category', 'Device_Type']
num_cols = ['Age', 'Transaction_Amount', 'Account_Balance', 'Gender']

def main():
<<<<<<< HEAD
    st.title('Fraud Detection App')

    # Input dari user
=======
    st.title('Fraud Detection Model')

    # User inputs
>>>>>>> 62a15c05933bc5872474dde66ca9553d569234f5
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

<<<<<<< HEAD
        # One-hot encoding
=======
        # One-hot encode categorical variables
>>>>>>> 62a15c05933bc5872474dde66ca9553d569234f5
        encoded_cats = pd.DataFrame(
            ohe.transform(df[cat_cols]),
            columns=ohe.get_feature_names_out(cat_cols),
            index=df.index
        )

<<<<<<< HEAD
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
=======
        # Combine numerical and encoded categorical features
        numerical_data = df[num_cols]
        df_final = pd.concat([numerical_data, encoded_cats], axis=1)

        # Match feature order with model
        model_features = model.get_booster().feature_names
        for col in model_features:
            if col not in df_final.columns:
                df_final[col] = 0  # Add missing columns as zero
        df_final = df_final[model_features]  # Reorder

        # Make prediction
        pred = model.predict(df_final)
        fraud_prediction = label_encoder.inverse_transform(pred)[0]

        # Optional: get fraud probability
        proba = model.predict_proba(df_final)[0][1]

        if fraud_prediction == 1:
            st.error(f"⚠️ Fraudulent transaction detected! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Legitimate transaction. (Probability of fraud: {proba:.2f})")
>>>>>>> 62a15c05933bc5872474dde66ca9553d569234f5

if __name__ == '__main__':
    main()
