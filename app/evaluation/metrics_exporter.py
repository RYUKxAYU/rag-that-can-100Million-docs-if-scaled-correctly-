import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class MetricsExporter:
    @staticmethod
    def export_json(metrics: Dict[str, Any], output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    @staticmethod
    def export_csv(records: Iterable[Dict[str, Any]], output_path: str) -> None:
        records = list(records)
        if not records:
            raise ValueError("No records provided for CSV export.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = sorted({key for record in records for key in record.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key, "") for key in headers})
