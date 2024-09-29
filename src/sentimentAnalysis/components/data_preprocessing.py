from sentimentAnalysis.config.configuration import DataPreprocessingConfig
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sentimentAnalysis.utils.common import save_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentimentAnalysis import logger
from sentimentAnalysis.utils.common import preprocess_text

class DataPreprocessingPipeline:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
    

    def main(self, data: pd.DataFrame):
        logger.info("Starting data preprocessing...")

        # Preprocess text data
        data["preprocessed_review"] = data["review"].apply(preprocess_text)

        # Encode sentiment labels
        data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

        # Split data into training and test sets
        train_data, test_data = train_test_split(data, test_size=self.config.train_test_split_ratio, random_state=self.config.random_state)

        save_data(train_data, self.config.train_data_path)
        save_data(test_data, self.config.test_data_path)

        # Tokenize text data for LSTM model
        # tokenizer = Tokenizer(num_words=self.config.max_words)
        # tokenizer.fit_on_texts(train_data["preprocessed_review"])
        # X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["preprocessed_review"]), maxlen=self.config.max_sequence_length)
        # X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["preprocessed_review"]), maxlen=self.config.max_sequence_length)

        
        # Use TF-IDF to vectorize text data for RFclassifier
        vectorizer = TfidfVectorizer(max_features=self.config.max_words)
        X_train = vectorizer.fit_transform(train_data["preprocessed_review"]).toarray()
        X_test = vectorizer.transform(test_data["preprocessed_review"]).toarray()

        logger.info("Data preprocessing completed successfully.")
        return X_train, X_test, train_data, test_data
