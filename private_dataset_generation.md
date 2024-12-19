# Mock Patient Dataset for Diabetes Chatbot

This document provides a detailed description of the mock dataset created to simulate realistic patient data for the
diabetes treatment chatbot. The dataset is designed to cover key demographics, lab results, symptoms, comorbidities,
treatments, and longitudinal data relevant to diabetes mellitus and its associated complications.

---

## **Dataset Description**

### **Columns**

1. **Demographics**
    - `patient_id` (Unique Identifier): A unique integer ID for each patient.
    - `age` (Integer): Patient's age in years.
    - `gender` (String): Gender of the patient (`Male` or `Female`).
    - `ethnicity` (String): Ethnic background (`Asian`, `Caucasian`, `African American`, etc.).
    - `pregnancy_status` (String, Nullable): Pregnancy status (`Pregnant`, `Not Pregnant`), applicable for female
      patients of childbearing age.
    - `weight_kg` (Float): Patient's weight in kilograms.
    - `height_cm` (Float): Patient's height in centimeters.
    - `bmi` (Float): Body Mass Index, calculated from weight and height.

2. **Lab Results**
    - `hba1c_percent` (Float): Glycated hemoglobin level (%).
    - `fasting_glucose_mg_dl` (Float): Fasting blood glucose level (mg/dL).
    - `postprandial_glucose_mg_dl` (Float, Nullable): Postprandial (after meal) glucose level (mg/dL).
    - `cholesterol_mg_dl` (Float): Total cholesterol level (mg/dL).
    - `hdl_mg_dl` (Float): High-density lipoprotein (good cholesterol) level (mg/dL).
    - `ldl_mg_dl` (Float): Low-density lipoprotein (bad cholesterol) level (mg/dL).
    - `triglycerides_mg_dl` (Float): Triglycerides level (mg/dL).
    - `blood_pressure_systolic_mm_hg` (Integer): Systolic blood pressure (mmHg).
    - `blood_pressure_diastolic_mm_hg` (Integer): Diastolic blood pressure (mmHg).
    - `kidney_function_gfr` (Float): Glomerular filtration rate (ml/min/1.73m²).

3. **Symptoms**
    - `symptoms` (String): Common symptoms reported by the patient (`Fatigue`, `Frequent urination`, `Blurred vision`,
      etc.).
    - `symptom_severity` (String): Severity level of symptoms (`Mild`, `Moderate`, `Severe`).

4. **Co-Morbidities**
    - `co_morbidities` (String): Existing co-morbid conditions (`Hypertension`, `Obesity`, `Coronary artery disease`,
      etc.).

5. **Treatments**
    - `current_medications` (String): Medications the patient is currently taking (`Metformin`, `Insulin`, etc.).
    - `treatment_history` (String): Summary of past treatments (`Lifestyle changes`, `Insulin therapy`, etc.).
    - `lifestyle_recommendations` (String, Nullable): Lifestyle advice provided (`Diet adjustment`, `Exercise routine`).

6. **Longitudinal Data**
    - `record_date` (Date): The date of the record.
    - `record_order` (Integer): Sequential order of the record for a given patient.

---

## **Example Dataset**

| patient_id | age | gender | ethnicity   | pregnancy_status | weight_kg | height_cm | bmi  | hba1c_percent | fasting_glucose_mg_dl | postprandial_glucose_mg_dl | cholesterol_mg_dl | hdl_mg_dl | ldl_mg_dl | triglycerides_mg_dl | blood_pressure_systolic_mm_hg | blood_pressure_diastolic_mm_hg | kidney_function_gfr | symptoms                    | symptom_severity | co_morbidities          | current_medications | treatment_history           | lifestyle_recommendations | record_date | record_order |
|------------|-----|--------|-------------|------------------|-----------|-----------|------|---------------|-----------------------|----------------------------|-------------------|-----------|-----------|---------------------|-------------------------------|--------------------------------|---------------------|-----------------------------|------------------|-------------------------|---------------------|-----------------------------|---------------------------|-------------|--------------|
| 1          | 45  | Male   | Caucasian   | N/A              | 80.0      | 175.0     | 26.1 | 7.8           | 145                   | 180                        | 200               | 50        | 120       | 150                 | 130                           | 85                             | 90.0                | Fatigue, Frequent urination | Moderate         | Hypertension, Obesity   | Metformin           | Lifestyle changes           | Diet adjustment           | 2024-01-01  | 1            |
| 2          | 32  | Female | African Am. | Pregnant         | 70.0      | 165.0     | 25.7 | 6.5           | 120                   | 140                        | 180               | 60        | 100       | 110                 | 120                           | 80                             | 100.0               | Fatigue, Blurred vision     | Mild             | None                    | Insulin             | Insulin therapy             | Exercise routine          | 2024-01-01  | 1            |
| 3          | 68  | Male   | Asian       | N/A              | 75.0      | 170.0     | 25.9 | 8.2           | 160                   | 200                        | 210               | 45        | 150       | 190                 | 140                           | 90                             | 60.0                | Frequent urination          | Severe           | Coronary artery disease | Insulin, Metformin  | Lifestyle + Insulin Therapy | None                      | 2024-01-01  | 1            |

---

This mock dataset includes realistic columns and values, ensuring the generated data aligns with clinical logic for the
diabetes treatment chatbot project.
