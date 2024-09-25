from sentimentAnalysis import logger
from sentimentAnalysis.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from sentimentAnalysis.components.data_validation import DataValidation
from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from sentimentAnalysis.pipeline.stage_03_model_building import ModelTrainingPipeline
import pandas as pd

STAGE_NAME= "Data Ingestion Stage"
STAGE_NAME_PREPROCESSING = "Data Preprocessing Stage"
STAGE_NAME_MODEL_TRAINING= "Model Building Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj= DataIngestionTrainingPipeline()
    data= obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

    # Data Validation
    data_validation_config = ConfigurationManager().get_data_validation_config()
    data_validator = DataValidation(config=data_validation_config)
    data_validator.validate_data(data)  # Pass the DataFrame to the validator
    logger.info("Data validation completed successfully. No issues found.")

     # Stage 2: Data Preprocessing
    logger.info(f">>>>>> stage {STAGE_NAME_PREPROCESSING} started <<<<<<")
    preprocessing_pipeline = DataPreprocessingTrainingPipeline()
    X_train, X_test, train_data, test_data = preprocessing_pipeline.main(data)
    logger.info(f">>>>>>> stage {STAGE_NAME_PREPROCESSING} completed <<<<<<<")

    # Model Training
    logger.info(f">>>>>> stage {STAGE_NAME_MODEL_TRAINING} started <<<<<<")
    model_training_pipeline = ModelTrainingPipeline()
    model = model_training_pipeline.main(X_train, train_data["sentiment"])  # Y_train would be the sentiment column
    logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_TRAINING} completed <<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e