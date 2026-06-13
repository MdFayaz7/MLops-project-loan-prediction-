from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


class ModelEvaluation:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        try:
            return self.model.predict(X)
        except Exception as e:
            raise Exception(f"[ModelEvaluation] Prediction failed: {e}")

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            print("Confusion Matrix:\n", cm)
            print("\n<------- Classification Report ------->\n")
            print(report)
            print("<------- Accuracy Scores ------->")
            print(f"Test Accuracy  : {acc:.4f}")
            print(f"Precision      : {precision:.4f}")
            print(f"Recall         : {recall:.4f}")

            return {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "confusion_matrix": cm,
                "classification_report": report,
            }
        except Exception as e:
            raise Exception(f"[ModelEvaluation] Evaluation failed: {e}")


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation
    from model_trainer import ModelTrainer

    ingestion = DataIngestion(filepath="loan_data.csv", target_column="not.fully.paid")
    df = ingestion.load_data()

    transformer = DataTransformation(categorical_columns=["purpose"])
    df = transformer.encode_categorical(df)

    X_train, X_test, y_train, y_test = ingestion.split_data(df)

    trainer = ModelTrainer(n_estimators=600)
    model = trainer.train(X_train, y_train)

    evaluator = ModelEvaluation(model=model)
    evaluator.evaluate(X_test, y_test)