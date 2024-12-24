import csv
import json
from pathlib import Path

# Define paths
CSV_FILE_PATH = Path("patient_data.csv")  # Input data file
BATCH_FILE_PATH = Path("batch_input.jsonl")  # Output batch file


def read_csv_data(csv_file_path):
    """
    Read data from a CSV file.

    :param csv_file_path: Path to the input CSV file.
    :return: List of dictionaries representing the CSV rows.
    """
    if not csv_file_path.exists():
        raise FileNotFoundError(f"Input CSV file {csv_file_path} does not exist.")

    with open(csv_file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        data = [row for row in reader]
    return data


def prepare_batch_file(patient_records, batch_file_path):
    """
    Prepare a .jsonl file for the Batch API.

    :param patient_records: List of patient records or input data.
    :param batch_file_path: Path to the output .jsonl file.
    """
    with open(batch_file_path, "w") as f:
        for idx, record in enumerate(patient_records):
            messages = build_openai_messages(record)
            batch_request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "max_tokens": OPENAI_MAX_TOKENS,
                    "temperature": OPENAI_TEMPERATURE,
                },
            }
            f.write(json.dumps(batch_request) + "\n")
    print(f"Batch file created at {batch_file_path}")


if __name__ == "__main__":
    # Read data from CSV
    patient_data = read_csv_data(CSV_FILE_PATH)

    # Prepare the batch file
    prepare_batch_file(patient_data, BATCH_FILE_PATH)
