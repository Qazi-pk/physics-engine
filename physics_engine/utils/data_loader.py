import csv
from pathlib import Path


def _coerce_value(value):
    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return None

    try:
        return float(text)
    except ValueError:
        return text


def load_csv(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        rows = []

        for row in reader:
            rows.append({key: _coerce_value(value) for key, value in row.items()})

    if not rows:
        raise ValueError("CSV file has no data rows")

    return headers, rows