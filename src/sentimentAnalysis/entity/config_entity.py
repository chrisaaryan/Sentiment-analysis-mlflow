from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir:  Path

@dataclass(frozen=True)
class DataValidationConfig:
    expected_columns: list
    sentiment_values: list

@dataclass(frozen=True)
class DataPreprocessingConfig:
    max_words: int
    max_sequence_length: int
    train_test_split_ratio: float
    random_state: int
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    input_dim: int
    output_dim: int
    input_length: int
    lstm_units: int
    dropout_rate: float
    recurrent_dropout_rate: float  
    optimizer: str
    learning_rate: float 
    loss: str
    metrics: List[str]
    epochs: int
    batch_size: int
    validation_split: float
    early_stopping_monitor: str  
    early_stopping_patience: int 
    save_model_path: Path

@dataclass(frozen=True)
class RFModelTrainingConfig:
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: str
    bootstrap: bool
    class_weight: str
    random_state: int
    save_model_path: Path


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    # test_data: Path
    all_params: dict
    mlflow_uri: str
