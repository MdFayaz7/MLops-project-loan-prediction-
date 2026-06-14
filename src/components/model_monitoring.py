import json
from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
from evidently import Report
# pyrefly: ignore [missing-import]
from evidently.presets import DataDriftPreset, DataSummaryPreset



from src.utils import load_config, ROOT_DIR

class ModelMonitoring:
    def __init__(self):
        self.config = load_config()
        self.reference_path = ROOT_DIR / self.config["data"]["raw_path"]
        self.current_path = ROOT_DIR / "artifacts" / "predictions.log.csv"
        self.report_html_path = ROOT_DIR / "artifacts" / "evidently_report.html"
        self.metrics_json_path = ROOT_DIR / "artifacts" / "monitoring_metrics.json"

    def run(self) -> bool:
        print("[ModelMonitoring] Starting monitoring run...")
        
        # 1. Load reference data
        if not self.reference_path.exists():
            print(f"[ModelMonitoring] Reference data not found at: {self.reference_path}")
            return False
        
        ref_df = pd.read_csv(self.reference_path)
        
        # 2. Load current prediction log data
        if not self.current_path.exists():
            print(f"[ModelMonitoring] Prediction log not found at: {self.current_path}")
            return False
        
        curr_df = pd.read_csv(self.current_path)
        
        if len(curr_df) < 2:
            print("[ModelMonitoring] Too few records in prediction log to evaluate drift (need at least 2).")
            return False

        # 3. Clean and align columns
        # Target column in reference data is 'not.fully.paid'
        target_col = self.config["data"]["target_column"]
        
        # Select features and target/prediction columns
        categorical_features = self.config["data"]["categorical_columns"]
        
        # Get numerical features from reference columns except target and categorical
        numerical_features = [
            col for col in ref_df.columns 
            if col != target_col and col not in categorical_features
        ]
        
        # Prepare reference dataframe
        ref_prepared = ref_df[numerical_features + categorical_features + [target_col]].copy()
        # Rename target column to 'prediction' in reference to align with prediction log if we want to compare distributions
        ref_prepared['prediction'] = ref_prepared[target_col]
        
        # Prepare current dataframe (only select corresponding columns)
        # Prediction log columns map to features and 'prediction'
        curr_prepared = curr_df[numerical_features + categorical_features + ['prediction']].copy()
        
        # Ensure numerical columns are floats, handle purpose as categorical
        for col in numerical_features:
            ref_prepared[col] = pd.to_numeric(ref_prepared[col], errors='coerce')
            curr_prepared[col] = pd.to_numeric(curr_prepared[col], errors='coerce')
            
        ref_prepared = ref_prepared.dropna(subset=numerical_features)
        curr_prepared = curr_prepared.dropna(subset=numerical_features)

        # 5. Generate Report
        report = Report(metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ])
        
        snapshot = report.run(
            reference_data=ref_prepared,
            current_data=curr_prepared
        )
        
        # 6. Save Reports
        ROOT_DIR.parent.mkdir(parents=True, exist_ok=True)
        self.report_html_path.parent.mkdir(parents=True, exist_ok=True)
        
        snapshot.save_html(str(self.report_html_path))
        print(f"[ModelMonitoring] HTML Report saved to: {self.report_html_path}")
        
        # Save key metrics summary to JSON
        metrics_dict = snapshot.dict()
        
        # Extract some high-level metrics for quick access
        try:
            drift_result = metrics_dict["metrics"][0]["value"]
            summary = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset_drift_detected": drift_result.get("dataset_drift", False),
                "drifted_features_ratio": drift_result.get("share_of_drifted_columns", 0.0),
                "number_of_drifted_features": drift_result.get("number_of_drifted_columns", 0),
                "total_features": drift_result.get("number_of_columns", 0)
            }
        except (KeyError, IndexError, TypeError) as e:
            print(f"[ModelMonitoring] Warning: could not parse exact drift metrics summary ({e}). Saving general snapshot info.")
            summary = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "snapshot_keys": list(metrics_dict.keys())
            }
        
        with open(self.metrics_json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[ModelMonitoring] Metrics summary saved to: {self.metrics_json_path}")
        
        return True

if __name__ == "__main__":
    monitoring = ModelMonitoring()
    monitoring.run()
