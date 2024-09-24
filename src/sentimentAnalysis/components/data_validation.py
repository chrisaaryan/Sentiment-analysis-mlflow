from sentimentAnalysis.config.configuration import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_data(self, data: pd.DataFrame):
        # Perform validation based on expected columns and sentiment values
        missing_columns = [col for col in self.config.expected_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        if not set(data['sentiment'].unique()).issubset(self.config.sentiment_values):
            raise ValueError("Invalid sentiment values.")
