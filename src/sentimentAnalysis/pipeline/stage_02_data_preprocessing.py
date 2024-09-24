from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.data_preprocessing import DataPreprocessingPipeline
from sentimentAnalysis import logger

STAGE_NAME_PREPROCESSING = "Data Preprocessing Stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self, data):
        try:
            # Fetch data preprocessing configuration
            data_preprocessing_config = ConfigurationManager().get_data_preprocessing_config()
            
            # Initialize the data preprocessing pipeline
            preprocessing_pipeline = DataPreprocessingPipeline(config=data_preprocessing_config)
            
            # Run the preprocessing pipeline on ingested data
            X_train, X_test, train_data, test_data = preprocessing_pipeline.main(data)
            return X_train, X_test, train_data, test_data

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME_PREPROCESSING} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME_PREPROCESSING} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
