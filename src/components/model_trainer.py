from sklearn.ensemble import RandomForestClassifier
import pickle


class ModelTrainer:
    def __init__(self, n_estimators: int = 600, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train) -> RandomForestClassifier:
        try:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)
            print(f"[ModelTrainer] Model trained with {self.n_estimators} estimators.")
            return self.model
        except Exception as e:
            raise Exception(f"[ModelTrainer] Training failed: {e}")

    def save_model(self, filepath: str = "model.pkl"):
        try:
            if self.model is None:
                raise ValueError("No trained model to save.")
            with open(filepath, "wb") as f:
                pickle.dump(self.model, f)
            print(f"[ModelTrainer] Model saved to {filepath}")
        except Exception as e:
            raise Exception(f"[ModelTrainer] Save failed: {e}")

    def load_model(self, filepath: str = "model.pkl"):
        try:
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
            print(f"[ModelTrainer] Model loaded from {filepath}")
            return self.model
        except FileNotFoundError:
            raise FileNotFoundError(f"[ModelTrainer] Model file not found: {filepath}")
        except Exception as e:
            raise Exception(f"[ModelTrainer] Load failed: {e}")


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_transformation import DataTransformation

    ingestion = DataIngestion(filepath="loan_data.csv", target_column="not.fully.paid")
    df = ingestion.load_data()

    transformer = DataTransformation(categorical_columns=["purpose"])
    df = transformer.encode_categorical(df)

    X_train, X_test, y_train, y_test = ingestion.split_data(df)

    trainer = ModelTrainer(n_estimators=600)
    trainer.train(X_train, y_train)
    trainer.save_model("model.pkl")