import os
from box.exceptions import BoxValueError
import yaml
import pandas as pd
from sentimentAnalysis import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
import re

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def preprocess_text(text: str) -> str:
    """Complete preprocessing for sentiment analysis."""
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    tokens = text.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    
    preprocessed_text = ' '.join(tokens)  # Return preprocessed text as a string

    if len(preprocessed_text) < 3:  # Length check
        raise ValueError("Invalid input.")
    return preprocessed_text
    # text = text.lower()  # Lowercase
    # text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    # text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    # tokens = text.split()  # Tokenize
    
    # # You could also handle contractions here

    # tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization

    # preprocessed_text = ' '.join(tokens)

    # if len(preprocessed_text) < 3:  # Length check
    #     raise ValueError("Invalid input.")
    
    # return preprocessed_text

def save_data(data: pd.DataFrame, path: str):
    """
    Save the given data (e.g., pandas DataFrame) to the specified path.

    Args:
        data (pd.DataFrame): The data to save (typically a pandas DataFrame).
        path (str): The path to save the data file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save data to CSV or other format as needed
        data.to_csv(path, index=False)

        logger.info(f"Data successfully saved to {path}.")

    except Exception as e:
        logger.exception(f"Error occurred while saving data to {path}: {e}")
        raise e

@ensure_annotations
def save_model(model, save_path):
    """
    Save the trained model to the specified path.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logger.info("Saving the model using joblib...")

        # Save the model using joblib
        joblib.dump(model, save_path)
        logger.info(f"Model saved successfully at {save_path}.")

    except Exception as e:
        logger.exception(f"Error occurred while saving the model: {e}")
        raise e
    
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a specified file path.

    Args:
        file_path (str): The path to the file to be loaded.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

