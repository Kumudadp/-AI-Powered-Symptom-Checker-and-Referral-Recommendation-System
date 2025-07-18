# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained components
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_encoder = joblib.load("symptom_encoder.pkl")  # MultiLabelBinarizer

# Get all symptoms from encoder
all_symptoms = list(symptom_encoder.classes_)

# Set up Streamlit UI
st.set_page_config(page_title="AI Symptom Checker", layout="centered")

st.title("🤖 AI Symptom Checker & Referral Recommendation System")
st.markdown("Select the symptoms you are experiencing:")

# Multiselect symptom input
selected_symptoms = st.multiselect("Symptoms", all_symptoms)

# Prepare input vector (1 row, multi-hot encoded)
input_symptom_array = symptom_encoder.transform([selected_symptoms])
input_df = pd.DataFrame(input_symptom_array, columns=symptom_encoder.classes_)

# Predict disease
if st.button("Check Diagnosis"):
    prediction = model.predict(input_df)[0]
    predicted_disease = label_encoder.inverse_transform([prediction])[0]

    st.success(f"🩺 Predicted Disease: **{predicted_disease}**")

    # Load full dataset to get recommendation
    df_info = pd.read_csv("AI_Symptom_Checker_Dataset.csv")
    df_info['Predicted Disease'] = df_info['Predicted Disease'].str.strip().str.lower()

    # Get matching disease row
    match = df_info[df_info['Predicted Disease'] == predicted_disease.lower()]

    if not match.empty:
        row = match.iloc[0]

        # Show correct specialist
        if 'Specialist' in row:
            specialist = row['Specialist']
            st.warning(f"👨‍⚕️ Consult a **{specialist}** for proper treatment.")

        # Show valid precautions
        if 'Precaution' in row:
            precaution = row['Precaution']
            st.info(f"🛡️ Recommended Care: {precaution}")

        # Optional: Add severity/confidence score if available
        if 'Severity' in row:
            st.info(f"⚠️ Severity: {row['Severity']}")
        if 'Confidence Score (%)' in row:
            st.info(f"📊 Confidence Score: {row['Confidence Score (%)']}%")
    else:
        st.error("No detailed info found for this disease. Please consult a doctor.")

