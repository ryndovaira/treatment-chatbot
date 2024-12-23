import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.generate_synthetic_data.config import (
    OUTPUT_FILE_BASIC_PATIENT_DATA,
    RANDOM_SEED,
)
from src.data.generate_synthetic_data.step_1_basic_patient_data.config import (
    LOG_FILE_NAME_VERIFICATION,
    PLOTS_DIR,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__, file_name=LOG_FILE_NAME_VERIFICATION)

np.random.seed(RANDOM_SEED)


# Load the synthetic data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading file at {file_path}: {e}")
        raise


# Save plots to the verification_results directory
def save_plot(plot, filename):
    try:
        plot_path = PLOTS_DIR / filename
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()  # Ensure plots are closed after saving
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")


# Check for missing data and log results
def check_missing_data(df):
    missing_data = df.isnull().sum()
    logger.warning("\nMissing Values per Column:\n" + missing_data.to_string())
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
    logger.info("\nVerification Results:")

    # Age-pregnancy relationship
    pregnancy_issues = df[
        (df["pregnancy_status"].notnull()) & ((df["age"] < 18) | (df["age"] > 45))
    ]
    if not pregnancy_issues.empty:
        logger.info(f"Pregnancy issues found:\n{pregnancy_issues}")
    else:
        logger.info("No issues with pregnancy status and age.")

    # BMI consistency
    df["calculated_bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    inconsistent_bmi = df[~(df["bmi"].round(1) == df["calculated_bmi"].round(1))]
    if not inconsistent_bmi.empty:
        logger.info(f"Inconsistent BMI records:\n{inconsistent_bmi}")
    else:
        logger.info("All BMI values are consistent.")

    # Gender-specific constraints
    gender_mismatch = df[(df["gender"] == "Male") & df["pregnancy_status"].notnull()]
    if not gender_mismatch.empty:
        logger.info(f"Gender mismatch found:\n{gender_mismatch}")
    else:
        logger.info("No gender mismatches.")


# Detect outliers
def detect_outliers(df):
    for col in ["hba1c_percent", "fasting_glucose_mg_dl", "blood_pressure_systolic_mm_hg"]:
        outliers = df[(df[col] < df[col].quantile(0.01)) | (df[col] > df[col].quantile(0.99))]
        if not outliers.empty:
            logger.info(f"Outliers detected in {col}:\n{outliers}")
        else:
            logger.info(f"No significant outliers in {col}.")


# Perform correlation checks
def check_correlations(df):
    correlations = df[["bmi", "blood_pressure_systolic_mm_hg", "fasting_glucose_mg_dl"]].corr()
    logger.info("\nCorrelations:\n" + correlations.to_string())


# Verify the synthetic data
def verify_synthetic_data():
    try:
        df = load_data(OUTPUT_FILE_BASIC_PATIENT_DATA)

        logger.info("Data Overview:\n" + df.head().to_string())
        logger.info("\nData Info:\n")
        logger.info(str(df.info()))

        # Perform checks and visualizations
        check_missing_data(df)
        plot_numeric_distributions(df)
        plot_categorical_distributions(df)
        verify_logical_constraints(df)
        detect_outliers(df)
        check_correlations(df)

        logger.info("Verification complete. Results logged.")
        print("Verification complete. Check logs for details.")
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        raise


if __name__ == "__main__":
    verify_synthetic_data()
