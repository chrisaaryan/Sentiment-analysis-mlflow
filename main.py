from sentimentAnalysis import logger
from sentimentAnalysis.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from sentimentAnalysis.components.data_validation import DataValidation
# from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from sentimentAnalysis.pipeline.stage_03_model_building import ModelTrainingPipeline
from sentimentAnalysis.pipeline.stage_04_model_eval import EvaluationPipeline

STAGE_NAME= "Data Ingestion Stage"
STAGE_NAME_PREPROCESSING = "Data Preprocessing Stage"
STAGE_NAME_MODEL_TRAINING= "Model Building Stage"
STAGE_NAME_MODEL_EVALUATION = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj= DataIngestionTrainingPipeline()
    data= obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")

    # # Data Validation
    
    # data_validator = DataValidation(config=data_validation_config)
    # data_validator.validate_data(data)  # Pass the DataFrame to the validator
    # logger.info("Data validation completed successfully. No issues found.")

    # Data Preprocessing
    logger.info(f">>>>>> stage {STAGE_NAME_PREPROCESSING} started <<<<<<")
    data_preprocessing_pipeline = DataPreprocessingTrainingPipeline()
    X_train, X_test, train_data, test_data = data_preprocessing_pipeline.main(data)
    logger.info(f">>>>>>> stage {STAGE_NAME_PREPROCESSING} completed <<<<<<<")

    # Model Training
    logger.info(f">>>>>> stage {STAGE_NAME_MODEL_TRAINING} started <<<<<<")
    model_training_pipeline = ModelTrainingPipeline()
    model = model_training_pipeline.main(X_train, train_data["sentiment"])  # Y_train would be the sentiment column
    logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_TRAINING} completed <<<<<<<")

    # Model Evaluation
    logger.info(f">>>>>> stage {STAGE_NAME_MODEL_EVALUATION} started <<<<<<")
    model_evaluation_pipeline = EvaluationPipeline()
    model_evaluation_pipeline.main(X_test, test_data["sentiment"])  # X_test and Y_test can be passed if needed within the evaluation stage
    logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_EVALUATION} completed <<<<<<<")

except Exception as e:
    logger.exception("An error occurred during the pipeline execution: ", exc_info=e)
    raise e