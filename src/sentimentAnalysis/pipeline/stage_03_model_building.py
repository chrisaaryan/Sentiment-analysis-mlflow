from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.model_builder import ModelBuilder
from sentimentAnalysis import logger
from sentimentAnalysis.utils.common import save_model

STAGE_NAME_MODEL_TRAINING = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self, X_train, Y_train):
        try:
            # Fetch model configuration from config
            config = ConfigurationManager()
            model_config = config.get_model_training_config()

            # Initialize the ModelBuilder with config parameters
            model_builder = ModelBuilder(
                input_dim=model_config.input_dim,
                output_dim=model_config.output_dim,
                input_length=model_config.input_length,
                lstm_units=model_config.lstm_units,
                dropout_rate=model_config.dropout_rate,
                optimizer=model_config.optimizer,
                loss=model_config.loss,
                metrics=model_config.metrics,
                epochs=model_config.epochs,              
                batch_size=model_config.batch_size,      
                validation_split=model_config.validation_split,
                save_model_path=model_config.save_model_path
            )

            # Build and compile the model
            model = model_builder.build_model()

            # Train the model
            logger.info("Starting model training...")
            model.fit(X_train, Y_train, epochs=model_config.epochs, batch_size=model_config.batch_size, validation_split=model_config.validation_split)
            logger.info("Model training completed successfully.")

            save_model(model, model_config.save_model_path)
            logger.info("Model saved successfully !")

            return model

        except Exception as e:
            logger.exception(f"Error occurred during model training: {e}")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME_MODEL_TRAINING} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()

        logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_TRAINING} completed <<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
