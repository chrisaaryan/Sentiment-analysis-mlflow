from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.data_preprocessing import DataPreprocessingPipeline
from sentimentAnalysis import logger
import os
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sentimentAnalysis.utils.common import load_data

STAGE_NAME_PREPROCESSING = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self, data):
        try:
            config = ConfigurationManager().get_data_preprocessing_config()
            train_data_path = config.train_data_path
            test_data_path = config.test_data_path
            if os.path.exists(train_data_path) and os.path.exists(test_data_path):
                logger.info("Loading preprocessed data from saved files...")
    
                # Load saved data
                train_data = load_data(train_data_path)
                test_data = load_data(test_data_path)
    
                # Tokenize the text data for LSTM
                # tokenizer = Tokenizer(num_words= config.max_words)
                # tokenizer.fit_on_texts(train_data["preprocessed_review"])
                # X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["preprocessed_review"]), maxlen= config.max_sequence_length)
                # X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["preprocessed_review"]), maxlen= config.max_sequence_length)

                # Use TF-IDF to vectorize text data for RFclassifier
                vectorizer = TfidfVectorizer(max_features= config.max_words)
                X_train = vectorizer.fit_transform(train_data["preprocessed_review"]).toarray()
                X_test = vectorizer.transform(test_data["preprocessed_review"]).toarray()
    
                logger.info("Preprocessed data loaded and tokenized successfully.")
                return X_train, X_test, train_data, test_data

            else:
                logger.info("Preprocessed data not found. Running data preprocessing pipeline...")
    
                # Run the preprocessing pipeline
                data_preprocessing_pipeline = DataPreprocessingPipeline(config= config)
                X_train, X_test, train_data, test_data = data_preprocessing_pipeline.main(data)

                logger.info("Data preprocessing completed and saved.")
                return X_train, X_test, train_data, test_data

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME_PREPROCESSING} started <<<<<<")
        data= load_data('artifacts/data_ingestion/IMDB Dataset.csv')
        obj = DataPreprocessingTrainingPipeline(data)
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME_PREPROCESSING} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
