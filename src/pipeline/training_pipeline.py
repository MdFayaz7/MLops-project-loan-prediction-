import mlflow
import mlflow.sklearn
import json
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils import load_config, ROOT_DIR


def run_training_pipeline():
    config = load_config()
    mlflow.set_experiment("loan-prediction")

    raw_path = ROOT_DIR / config["data"]["raw_path"]
    processed_path = ROOT_DIR / config["data"]["processed_path"]
    target = config["data"]["target_column"]

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params(config["model"])
        mlflow.log_param("target", target)

        # 1. Ingest
        ingestion = DataIngestion(
            filepath=str(raw_path),
            target_column=target,
            test_size=config["model"]["test_size"],
        )
        df = ingestion.load_data()

        # 2. Validate
        validation = DataValidation(
            required_columns=list(df.columns),
            target_column=target,
        )
        validation.validate(df)

        # 3. Transform
        transformer = DataTransformation(
            categorical_columns=config["data"]["categorical_columns"]
        )
        df = transformer.encode_categorical(df)
        df.to_csv(processed_path, index=False)
        print(f"[TrainingPipeline] Processed data saved to {processed_path}")

        # 4. Split
        X_train, X_test, y_train, y_test = ingestion.split_data(df)

        # 5. Train
        trainer = ModelTrainer(n_estimators=config["model"]["n_estimators"])
        model = trainer.train(X_train, y_train)
        trainer.save_model(str(ROOT_DIR / config["artifacts"]["model_path"]))

        # 6. Evaluate
        evaluator = ModelEvaluation(model=model)
        metrics = evaluator.evaluate(X_test, y_test)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.sklearn.log_model(model, "model")
        run_id=mlflow.active_run().info.run_id
        version_file = ROOT_DIR / "artifacts" / "model_version.json"
        with open(version_file,"w") as f:
          json.dump({"model_version":run_id},f)


if __name__ == "__main__":
    run_training_pipeline()
