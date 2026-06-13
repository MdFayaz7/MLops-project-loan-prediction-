import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataTransformation:
    def __init__(self, categorical_columns: list = None, scale_features: bool = False):
        self.categorical_columns = categorical_columns or []
        self.scale_features = scale_features
        self.label_encoders = {}
        self.scaler = StandardScaler() if scale_features else None

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for col in self.categorical_columns:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not found for encoding.")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"[DataTransformation] Encoded column: {col}")
            return df
        except Exception as e:
            raise Exception(f"[DataTransformation] Encoding failed: {e}")

    def scale(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        try:
            if self.scaler is None:
                return X_train, X_test
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            print("[DataTransformation] Scaling applied.")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            raise Exception(f"[DataTransformation] Scaling failed: {e}")

    def transform(self, df: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame):
        df = self.encode_categorical(df)
        X_train, X_test = self.scale(X_train, X_test)
        print("[DataTransformation] Transformation complete.")
        return df, X_train, X_test


if __name__ == "__main__":
    from data_ingestion import DataIngestion

    ingestion = DataIngestion(filepath="loan_data.csv", target_column="not.fully.paid")
    df = ingestion.load_data()

    transformer = DataTransformation(categorical_columns=["purpose"])
    df = transformer.encode_categorical(df)

    X_train, X_test, y_train, y_test = ingestion.split_data(df)