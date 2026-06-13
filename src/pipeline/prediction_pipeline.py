import json

import pandas as pd

from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation
from src.utils import load_config, ROOT_DIR


def get_model_version() -> str:
    version_file = ROOT_DIR / "artifacts" / "model_version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)["model_version"]
    return "unknown"


def run_prediction_pipeline(input_data: dict) -> tuple[int, str]:
    config = load_config()

    trainer = ModelTrainer()
    model = trainer.load_model(str(ROOT_DIR / config["artifacts"]["model_path"]))

    df = pd.DataFrame([input_data])
    transformer = DataTransformation(
        categorical_columns=config["data"]["categorical_columns"]
    )
    df = transformer.encode_categorical(df)

    evaluator = ModelEvaluation(model=model)
    prediction = int(evaluator.predict(df)[0])
    return prediction, get_model_version()
