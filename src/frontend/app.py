import os

import requests
import streamlit as st

from src.logging_config import setup_logger
from src.patient_data_params import *

logger = setup_logger(__name__)

API_URL = f"{os.getenv("BACKEND_API_URL", "http://localhost:8000")}/query"

st.title("Diabetes Treatment Support Chatbot")


base_query = st.text_input("Query", placeholder="What is the recommended treatment?")

patient_data_toggle = st.toggle("Patient Data")
if patient_data_toggle:
    age = st.number_input("Age", min_value=AGE_RANGE[0], max_value=AGE_RANGE[1], step=1)
    gender = st.selectbox("Gender", GENDERS)
    height_cm = st.number_input(
        "Height (cm)", min_value=HEIGHT_CM_RANGE[0], max_value=HEIGHT_CM_RANGE[1]
    )
    weight_kg = st.number_input(
        "Weight (kg)", min_value=WEIGHT_KG_RANGE[0], max_value=WEIGHT_KG_RANGE[1]
    )
    bmi = compute_bmi(weight_kg, height_cm)
    st.number_input("BMI", value=bmi, format="%.1f", key="bmi", disabled=True)
    ethnicity = st.selectbox("Ethnicity", ETHNICITIES)
    pregnancy_status = st.selectbox("Pregnancy Status", PREGNANCY_STATUS)
    symptoms = st.multiselect("Symptoms", SYMPTOMS)
    severity = st.selectbox("Symptom Severity", SYMPTOM_SEVERITY)
    co_morbidities = st.multiselect("Co-morbidities", CO_MORBIDITIES)
else:
    age = None
    gender = None
    height_cm = None
    weight_kg = None
    bmi = None
    ethnicity = None
    pregnancy_status = None
    symptoms = None
    severity = None
    co_morbidities = None

lab_results_toggle = st.toggle("Lab Results")

if lab_results_toggle:
    hba1c_percent = st.number_input(
        "HbA1c (%)",
        min_value=LAB_RANGES["hba1c_percent"][0],
        max_value=LAB_RANGES["hba1c_percent"][1],
        format="%.1f",
    )
    fasting_glucose_mg_dl = st.number_input(
        "Fasting Glucose (mg/dL)",
        min_value=LAB_RANGES["fasting_glucose_mg_dl"][0],
        max_value=LAB_RANGES["fasting_glucose_mg_dl"][1],
        format="%.1f",
    )
    postprandial_glucose_mg_dl = st.number_input(
        "Postprandial Glucose (mg/dL)",
        min_value=LAB_RANGES["postprandial_glucose_mg_dl"][0],
        max_value=LAB_RANGES["postprandial_glucose_mg_dl"][1],
        format="%.1f",
    )
    cholesterol_mg_dl = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=LAB_RANGES["cholesterol_mg_dl"][0],
        max_value=LAB_RANGES["cholesterol_mg_dl"][1],
        format="%.1f",
    )
    hdl_mg_dl = st.number_input(
        "HDL (mg/dL)",
        min_value=LAB_RANGES["hdl_mg_dl"][0],
        max_value=LAB_RANGES["hdl_mg_dl"][1],
        format="%.1f",
    )
    ldl_mg_dl = st.number_input(
        "LDL (mg/dL)",
        min_value=LAB_RANGES["ldl_mg_dl"][0],
        max_value=LAB_RANGES["ldl_mg_dl"][1],
        format="%.1f",
    )
    triglycerides_mg_dl = st.number_input(
        "Triglycerides (mg/dL)",
        min_value=LAB_RANGES["triglycerides_mg_dl"][0],
        max_value=LAB_RANGES["triglycerides_mg_dl"][1],
        format="%.1f",
    )
    blood_pressure_systolic_mm_hg = st.number_input(
        "Blood Pressure Systolic (mmHg)",
        min_value=LAB_RANGES["blood_pressure_systolic_mm_hg"][0],
        max_value=LAB_RANGES["blood_pressure_systolic_mm_hg"][1],
        format="%.1f",
    )
    blood_pressure_diastolic_mm_hg = st.number_input(
        "Blood Pressure Diastolic (mmHg)",
        min_value=LAB_RANGES["blood_pressure_diastolic_mm_hg"][0],
        max_value=LAB_RANGES["blood_pressure_diastolic_mm_hg"][1],
        format="%.1f",
    )
    kidney_function_gfr = st.number_input(
        "Kidney Function GFR (mL/min/1.73m^2)",
        min_value=LAB_RANGES["kidney_function_gfr"][0],
        max_value=LAB_RANGES["kidney_function_gfr"][1],
        format="%.1f",
    )
else:
    hba1c_percent = None
    fasting_glucose_mg_dl = None
    postprandial_glucose_mg_dl = None
    cholesterol_mg_dl = None
    hdl_mg_dl = None
    ldl_mg_dl = None
    triglycerides_mg_dl = None
    blood_pressure_systolic_mm_hg = None
    blood_pressure_diastolic_mm_hg = None
    kidney_function_gfr = None


if st.button("Submit"):
    patient_data = {
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "ethnicity": ethnicity,
        "pregnancy_status": pregnancy_status,
        "symptoms": symptoms,
        "severity": severity,
        "co_morbidities": co_morbidities,
        "hba1c_percent": hba1c_percent,
        "fasting_glucose_mg_dl": fasting_glucose_mg_dl,
        "postprandial_glucose_mg_dl": postprandial_glucose_mg_dl,
        "cholesterol_mg_dl": cholesterol_mg_dl,
        "hdl_mg_dl": hdl_mg_dl,
        "ldl_mg_dl": ldl_mg_dl,
        "triglycerides_mg_dl": triglycerides_mg_dl,
        "blood_pressure_systolic_mm_hg": blood_pressure_systolic_mm_hg,
        "blood_pressure_diastolic_mm_hg": blood_pressure_diastolic_mm_hg,
        "kidney_function_gfr": kidney_function_gfr,
    }
    logger.info(f"Patient data: {patient_data}")

    patient_data = {k: v for k, v in patient_data.items() if v is not None}
    logger.info(f"Patient data after removing None values: {patient_data}")

    response = requests.post(API_URL, json={"patient_data": patient_data, "base_query": base_query})
    logger.info(f"API response: {response.json()}")

    if response.status_code == 200:
        data = response.json()
        st.header("Results")
        st.subheader("Public Summary")
        st.write(data["public_summary"])
        st.subheader("Private Summary")
        st.write(data["private_summary"])
        st.subheader("Combined Summary")
        st.write(data["combined_summary"])
    else:
        st.error("Error processing query.")
