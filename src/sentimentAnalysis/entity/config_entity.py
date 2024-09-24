from dataclasses import dataclass
from pathlib import Path

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
