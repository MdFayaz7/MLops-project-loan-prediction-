import csv
from datetime import datetime
from pathlib import Path
LOG_FILE = Path("artifacts/predictions.log.csv")
def log_prediction(input_data: dict, prediction: int,model_version:str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp","model_version", "prediction", "label", *input_data.keys()],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "model_version":model_version,
            "prediction": prediction,
            "label": "Approved" if prediction == 0 else "Rejected",
            **input_data,
        })
