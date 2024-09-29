from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.model_eval import Evaluation
from sentimentAnalysis import logger

STAGE_NAME_MODEL_EVALUATION = "Model Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self, X_test, Y_test):
        try:
            # Fetch evaluation configuration
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()

            # Initialize the Evaluation class with config parameters
            evaluation = Evaluation(eval_config)
            logger.info("Starting model evaluation...")
            evaluation.evaluation(X_test, Y_test)  # Evaluate the model
            evaluation.log_into_mlflow()  # Log metrics and model to MLflow
            logger.info("Model evaluation and logging completed successfully.")

        except Exception as e:
            logger.exception(f"Error occurred during model evaluation: {e}")
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME_MODEL_EVALUATION} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()

        logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_EVALUATION} completed <<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
