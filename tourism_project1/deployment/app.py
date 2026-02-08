import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Ensure the directory exists
os.makedirs("/content/tourism_project1/deployment", exist_ok=True)

# Download and load the model
model_path = hf_hub_download(repo_id="Praveen-AISRM/Tourism-Package-Prediction", filename="best_Tourism_Package_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Booking App Prediction
st.title("Wellness Package-Tourism Booking App")
st.write("""
This application predicts the likelihood of Booking the Tourism Package that is introduced by the Tourism App  based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Single","Divorced","Unmarried"])
Type = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)
MonthlyIncome = st.number_input( "Monthly Income (₹)",    min_value=15000,    max_value=40000,    value=20000,    step=500)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=22, value=10)
ProductPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic","Standard","King","Super Deluxe"])
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127, value=80)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting",  min_value=1, max_value=5, value=2)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=5, value=1)
Gender = st.selectbox("Gender", ["Male", "Female"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': Type,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': "Executive"  # default value (required)
}])


if st.button("Predict Wellness Package-Tourism Booking"):
    prediction = model.predict(input_data)[0]
    result = "Booking Done" if prediction == 1 else "No Booking"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
