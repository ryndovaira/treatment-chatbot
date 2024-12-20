from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.generate_synthetic_data.synthetic_data_config import (
    OUTPUT_FILE,
    RANDOM_SEED,
    LOG_FILE_V,
    PLOTS_DIR,
)

np.random.seed(RANDOM_SEED)


# Log verification results with timestamps
def log_results(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    with open(LOG_FILE_V, "a") as log_file:
        log_file.write(full_message + "\n")


# Load the synthetic data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        log_results(f"Error loading file at {file_path}: {e}")
        raise


# Save plots to the verification_results directory
def save_plot(plot, filename):
    try:
        plot_path = PLOTS_DIR / filename
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()  # Ensure plots are closed after saving
        log_results(f"Plot saved to {plot_path}")
    except Exception as e:
        log_results(f"Error saving plot {filename}: {e}")


# Check for missing data and log results
def check_missing_data(df):
    missing_data = df.isnull().sum()
    log_results("\nMissing Values per Column:\n" + missing_data.to_string())
    return missing_data


# Check numeric column distributions
def plot_numeric_distributions(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid()
        save_plot(plt, f"{col}_distribution.png")


# Check categorical column distributions
def plot_categorical_distributions(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Frequency")
        plt.ylabel(col)
        plt.grid()
        save_plot(plt, f"{col}_distribution.png")


# Verify logical consistency
def verify_logical_constraints(df):
    log_results("\nVerification Results:")

    # Age-pregnancy relationship
    pregnancy_issues = df[
        (df["pregnancy_status"].notnull()) & ((df["age"] < 18) | (df["age"] > 45))
    ]
    if not pregnancy_issues.empty:
        log_results(f"Pregnancy issues found:\n{pregnancy_issues}")
    else:
        log_results("No issues with pregnancy status and age.")

    # BMI consistency
    df["calculated_bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    inconsistent_bmi = df[~(df["bmi"].round(1) == df["calculated_bmi"].round(1))]
    if not inconsistent_bmi.empty:
        log_results(f"Inconsistent BMI records:\n{inconsistent_bmi}")
    else:
        log_results("All BMI values are consistent.")

    # Gender-specific constraints
    gender_mismatch = df[(df["gender"] == "Male") & df["pregnancy_status"].notnull()]
    if not gender_mismatch.empty:
        log_results(f"Gender mismatch found:\n{gender_mismatch}")
    else:
        log_results("No gender mismatches.")


# Detect outliers
def detect_outliers(df):
    for col in ["hba1c_percent", "fasting_glucose_mg_dl", "blood_pressure_systolic_mm_hg"]:
        outliers = df[(df[col] < df[col].quantile(0.01)) | (df[col] > df[col].quantile(0.99))]
        if not outliers.empty:
            log_results(f"Outliers detected in {col}:\n{outliers}")
        else:
            log_results(f"No significant outliers in {col}.")


# Perform correlation checks
def check_correlations(df):
    correlations = df[["bmi", "blood_pressure_systolic_mm_hg", "fasting_glucose_mg_dl"]].corr()
    log_results("\nCorrelations:\n" + correlations.to_string())


# Verify the synthetic data
def verify_synthetic_data():
    try:
        df = load_data(OUTPUT_FILE)

        # Log basic info
        log_results("Data Overview:\n" + df.head().to_string())
        log_results("\nData Info:\n")
        log_results(str(df.info()))

        # Perform checks and visualizations
        check_missing_data(df)
        plot_numeric_distributions(df)
        plot_categorical_distributions(df)
        verify_logical_constraints(df)
        detect_outliers(df)
        check_correlations(df)

        log_results("Verification complete. Results logged.")
        print("Verification complete. Check logs for details.")
    except Exception as e:
        log_results(f"Error during verification: {e}")
        raise


if __name__ == "__main__":
    verify_synthetic_data()
