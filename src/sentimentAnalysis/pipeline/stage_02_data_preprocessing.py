from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.data_preprocessing import DataPreprocessingPipeline
from sentimentAnalysis import logger
import os
import pandas as pd
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sentimentAnalysis.utils.common import load_data

STAGE_NAME_PREPROCESSING = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self, data: pd.DataFrame):
        self.data= data

    def main(self):
        try:
            config = ConfigurationManager().get_data_preprocessing_config()
            # train_data_path = config.train_data_path
            # test_data_path = config.test_data_path
            # if os.path.exists(train_data_path) and os.path.exists(test_data_path):
            #     logger.info("Loading preprocessed data from saved files...")
    
            #     # Load saved data
            #     train_data = load_data(train_data_path)
            #     test_data = load_data(test_data_path)
    
            #     # Tokenize the text data for LSTM
            #     # tokenizer = Tokenizer(num_words= config.max_words)
            #     # tokenizer.fit_on_texts(train_data["preprocessed_review"])
            #     # X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["preprocessed_review"]), maxlen= config.max_sequence_length)
            #     # X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["preprocessed_review"]), maxlen= config.max_sequence_length)

            #     # Use TF-IDF to vectorize text data for RFclassifier
            #     vectorizer = TfidfVectorizer(max_features= config.max_words)
            #     X_train = vectorizer.fit_transform(train_data["preprocessed_review"]).toarray()
            #     X_test = vectorizer.transform(test_data["preprocessed_review"]).toarray()
    
            #     logger.info("Preprocessed data loaded and tokenized successfully.")
            #     return X_train, X_test, train_data, test_data

            # else:
            logger.info("Running data preprocessing pipeline...")

            # Ensure the output directory exists
            os.makedirs('artifacts/data_preprocessing', exist_ok=True)
    
            # Run the preprocessing pipeline
            data_preprocessing_pipeline = DataPreprocessingPipeline(config= config)
            X_train, X_test, train_data, test_data = data_preprocessing_pipeline.main(data)
            pd.DataFrame(X_train).to_csv('artifacts/data_preprocessing/X_train.csv', index=False)
            pd.DataFrame(X_test).to_csv('artifacts/data_preprocessing/X_test.csv', index=False)
            pd.DataFrame(train_data).to_csv('artifacts/data_preprocessing/train_data.csv', index=False)
            pd.DataFrame(test_data).to_csv('artifacts/data_preprocessing/test_data.csv', index=False)


            logger.info("Data preprocessing completed and saved.")
            return X_train, X_test, train_data, test_data

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME_PREPROCESSING} started <<<<<<")
        data= load_data('artifacts/data_ingestion/IMDB_Dataset.csv')
        data_preprocessing = DataPreprocessingTrainingPipeline(data)
        X_train, X_test, train_data, test_data = data_preprocessing.main()
        logger.info(f">>>>>>> stage {STAGE_NAME_PREPROCESSING} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
