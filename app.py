import streamlit as st 

# Set Streamlit page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction Tool",
    page_icon="❤️",
    layout="centered"
)

# Add logo and title
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Heart_icon_red_hollow.svg/1024px-Heart_icon_red_hollow.svg.png", width=100)
st.markdown("<h1 style='text-align: center; color: red;'>Heart Attack Risk Prediction Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>This AI-powered tool helps patients and doctors assess the likelihood of heart attack risk based on medical parameters.</p>", unsafe_allow_html=True)

# ✅ Your existing imports and code come after this:
import pandas as pd
import joblib
# ... rest of your app code continues here ...

import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("final_heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Prediction")

st.write("Please fill in the following medical information to assess heart disease risk.")

def user_input_features():
    age = st.number_input('Age (years)', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Gender', ['Male', 'Female'])
    chest_pain_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    resting_blood_pressure = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
    serum_cholesterol = st.number_input('Serum Cholesterol (mg/dL)', min_value=100, max_value=600, value=200)
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250, value=150)
    oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    thal = st.selectbox('Thalassemia Type', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    sex_val = 1 if sex == 'Male' else 0
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

    data = {
        'age': age,
        'sex': sex_val,
        'chest_pain_type': cp_map[chest_pain_type],
        'resting_blood_pressure': resting_blood_pressure,
        'serum_cholesterol': serum_cholesterol,
        'thalach': thalach,
        'oldpeak': oldpeak,
        'thal': thal_map[thal]
    }
    features_df = pd.DataFrame([data])
    return features_df

input_df = user_input_features()

if st.button('Predict'):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"🔴 High risk of heart disease! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low risk of heart disease. Probability: {probability:.2%}")
