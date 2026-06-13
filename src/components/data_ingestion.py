import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, filepath: str, target_column: str, test_size: float = 0.3, random_state: int = 101):
        self.filepath = filepath
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.filepath)
            print(f"[DataIngestion] Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"[DataIngestion] File not found at path: {self.filepath}")
        except Exception as e:
            raise Exception(f"[DataIngestion] Error loading data: {e}")

    def split_data(self, df: pd.DataFrame):
        try:
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            print(f"[DataIngestion] Train size: {X_train.shape}, Test size: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except KeyError:
            raise KeyError(f"[DataIngestion] Target column '{self.target_column}' not found in DataFrame.")
        except Exception as e:
            raise Exception(f"[DataIngestion] Error splitting data: {e}")


if __name__ == "__main__":
    ingestion = DataIngestion(filepath="loan_data.csv", target_column="not.fully.paid")
    df = ingestion.load_data()
    X_train, X_test, y_train, y_test = ingestion.split_data(df)