import json
from pathlib import Path


def validate_metadata(directory_path):
    """
    Validates metadata.json against the files in the directory.

    :param directory_path: Path to the directory containing metadata.json and files.
    """
    metadata_path = directory_path / "metadata.json"

    # Load metadata.json
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: {metadata_path} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding {metadata_path}: {e}")
        return

    # Ensure metadata structure is valid
    try:
        files_in_metadata = {entry["file_name"] for entry in metadata if "file_name" in entry}
    except KeyError as e:
        print(f"Error: Metadata entry is missing a required key: {e}")
        return

    # Get actual files in the directory
    files_in_directory = {file.name for file in directory_path.glob("*.pdf")}

    # Check for missing files
    missing_files = files_in_metadata - files_in_directory
    if missing_files:
        print("Missing files (present in metadata.json but not in directory):")
        for file in missing_files:
            print(f" - {file}")
    else:
        print("No missing files in directory.")

    # Check for extra files
    extra_files = files_in_directory - files_in_metadata
    if extra_files:
        print("Extra files (present in directory but not in metadata.json):")
        for file in extra_files:
            print(f" - {file}")
    else:
        print("No extra files in directory.")

    # Final summary
    if not missing_files and not extra_files:
        print(f"Validation successful for {directory_path}!")
    else:
        print(f"Validation completed for {directory_path} with discrepancies.")


def main():
    # Paths are relative to the project structure
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"

    public_directory = raw_data_dir / "public"
    private_directory = raw_data_dir / "private"

    print("Validating public directory...")
    validate_metadata(public_directory)

    print("\nValidating private directory...")
    validate_metadata(private_directory)


if __name__ == "__main__":
    main()
