import matplotlib.pyplot as plt
# Set random seed for reproducibility in any additional logic
import numpy as np
import pandas as pd
import seaborn as sns

from synthetic_data_config import OUTPUT_FILE, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# Define output directory for plots
PLOTS_DIR = OUTPUT_FILE.parent.parent / "private" / "verification_results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Load the synthetic data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading file at {file_path}: {e}")


# Save plots to the verification_results directory
def save_plot(plot, filename):
    plot_path = PLOTS_DIR / filename
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


# Generate visualizations and checks
def verify_synthetic_data():
    # Load data
    df = load_data(OUTPUT_FILE)

    # Quick overview
    print("Data Overview:")
    print(df.head())
    print("\nData Info:")
    print(df.info())

    # Check for missing values
    missing_data = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing_data)

    # Distributions of numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid()
        save_plot(plt, f"{col}_distribution.png")
        plt.close()

    # Categorical data distributions
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Frequency")
        plt.ylabel(col)
        plt.grid()
        save_plot(plt, f"{col}_distribution.png")
        plt.close()


if __name__ == "__main__":
    verify_synthetic_data()
