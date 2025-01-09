import json
from pathlib import Path

import pandas as pd


def load_json_data(file_path: Path) -> list:
    with open(file_path, "r") as f:
        return json.load(f)


def get_unique_patient_ids(data: list) -> set:
    return {record["patient_id"] for record in data}


def load_csv_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def print_record_count(data: list) -> None:
    print("Number of records in patient data and treatment:", len(data))


def print_unique_patient_ids_count(patient_ids: set) -> None:
    print("Number of unique patient IDs in patient data and treatment:", len(patient_ids))


def print_csv_unique_patient_ids_count(data: pd.DataFrame) -> None:
    print("Number of unique patient IDs in basic patient data:", data.patient_id.nunique())


def get_patient_data_and_treatment_path() -> Path:
    patient_data_and_treatment_file_name = "patient_data_and_treatment_6.json"
    return (
        Path(__file__).resolve().parent
        / "openai_async"
        / "artifacts"
        / patient_data_and_treatment_file_name
    )


def get_basic_patient_data_path() -> Path:

    basic_patient_data_file_name = (
        Path(__file__).resolve().parents[4] / "data" / "raw" / "private" / "basic_patient_data.csv"
    )
    return Path(__file__).resolve().parent / basic_patient_data_file_name


def main():
    patient_data_and_treatment_path = get_patient_data_and_treatment_path()
    data = load_json_data(patient_data_and_treatment_path)
    print_record_count(data)

    patient_ids = get_unique_patient_ids(data)
    print_unique_patient_ids_count(patient_ids)

    basic_patient_data_path = get_basic_patient_data_path()
    data = load_csv_data(basic_patient_data_path)
    print_csv_unique_patient_ids_count(data)


if __name__ == "__main__":
    main()
