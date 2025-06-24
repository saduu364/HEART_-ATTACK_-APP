
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("final_heart_disease_model.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Please enter the patient's features (all values are standardized):")

def user_input_features():
    age = st.slider('Age (standardized)', -3.0, 3.0, 0.0)
    sex = st.slider('Sex (standardized)', -2.0, 2.0, 0.0)
    chest_pain_type = st.slider('Chest Pain Type (standardized)', -2.0, 3.0, 0.0)
    resting_blood_pressure = st.slider('Resting Blood Pressure (standardized)', -3.0, 3.0, 0.0)
    serum_cholesterol = st.slider('Serum Cholesterol (standardized)', -3.0, 3.0, 0.0)
    fasting_blood_sugar = st.slider('Fasting Blood Sugar (standardized)', -3.0, 3.0, 0.0)
    restecg = st.slider('Rest ECG (standardized)', -3.0, 3.0, 0.0)
    thalach = st.slider('Max Heart Rate (standardized)', -3.0, 3.0, 0.0)
    exang = st.slider('Exercise Induced Angina (standardized)', -3.0, 3.0, 0.0)
    oldpeak = st.slider('ST Depression (standardized)', -3.0, 3.0, 0.0)
    slope = st.slider('Slope (standardized)', -3.0, 3.0, 0.0)
    ca = st.slider('Number of Major Vessels (standardized)', -3.0, 3.0, 0.0)
    thal = st.slider('Thalassemia (standardized)', -3.0, 3.0, 0.0)

    data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'serum_cholesterol': serum_cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features_df = pd.DataFrame([data])
    return features_df

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease! Probability: {proba:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease! Probability: {proba:.2f}")
