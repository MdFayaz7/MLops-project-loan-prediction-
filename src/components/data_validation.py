import pandas as pd


class DataValidation:
    def __init__(self, required_columns: list, target_column: str):
        self.required_columns = required_columns
        self.target_column = target_column

    def check_missing_columns(self, df: pd.DataFrame):
        try:
            missing = [col for col in self.required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"[DataValidation] Missing columns: {missing}")
            print("[DataValidation] All required columns present.")
        except Exception as e:
            raise Exception(f"[DataValidation] Column check failed: {e}")

    def check_null_values(self, df: pd.DataFrame):
        try:
            null_counts = df.isnull().sum()
            nulls = null_counts[null_counts > 0]
            if not nulls.empty:
                print(f"[DataValidation] Null values found:\n{nulls}")
            else:
                print("[DataValidation] No null values found.")
        except Exception as e:
            raise Exception(f"[DataValidation] Null check failed: {e}")

    def check_target_distribution(self, df: pd.DataFrame):
        try:
            dist = df[self.target_column].value_counts()
            print(f"[DataValidation] Target distribution:\n{dist}")
        except KeyError:
            raise KeyError(f"[DataValidation] Target column '{self.target_column}' not found.")
        except Exception as e:
            raise Exception(f"[DataValidation] Target check failed: {e}")

    def validate(self, df: pd.DataFrame):
        self.check_missing_columns(df)
        self.check_null_values(df)
        self.check_target_distribution(df)
        print("[DataValidation] Validation complete.")


if __name__ == "__main__":
    from data_ingestion import DataIngestion

    ingestion = DataIngestion(filepath="loan_data.csv", target_column="not.fully.paid")
    df = ingestion.load_data()

    required_cols = list(df.columns)
    validation = DataValidation(required_columns=required_cols, target_column="not.fully.paid")
    validation.validate(df)