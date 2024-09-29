# from sentimentAnalysis.config.configuration import ConfigurationManager
# from sentimentAnalysis.components.model_builder import ModelBuilder
# from sentimentAnalysis import logger
# from sentimentAnalysis.utils.common import save_model

# STAGE_NAME_MODEL_TRAINING = "Model Training Stage"

# class ModelTrainingPipeline:
#     def __init__(self):
#         pass

#     def main(self, X_train, Y_train):
#         try:
#             # Fetch model configuration from config
#             config = ConfigurationManager()
#             model_config = config.get_model_training_config()

#             # Initialize the ModelBuilder with config parameters
#             model_builder = ModelBuilder(
#                 input_dim=model_config.input_dim,
#                 output_dim=model_config.output_dim,
#                 input_length=model_config.input_length,
#                 lstm_units=model_config.lstm_units,
#                 dropout_rate=model_config.dropout_rate,
#                 recurrent_dropout_rate=model_config.recurrent_dropout_rate,  # Added
#                 optimizer=model_config.optimizer,
#                 learning_rate=model_config.learning_rate,  # Added
#                 loss=model_config.loss,
#                 metrics=model_config.metrics,
#                 epochs=model_config.epochs,              
#                 batch_size=model_config.batch_size,      
#                 validation_split=model_config.validation_split,
#                 early_stopping_monitor=model_config.early_stopping_monitor,  # Added
#                 early_stopping_patience=model_config.early_stopping_patience,  # Added
#                 save_model_path=model_config.save_model_path
#             )

#             # Build and compile the model
#             model = model_builder.build_model()

#             # Get early stopping callback
#             early_stopping = model_builder.get_early_stopping_callback()

#             # Train the model with early stopping
#             logger.info("Starting model training...")
#             model.fit(X_train, Y_train, epochs=model_config.epochs, batch_size=model_config.batch_size, validation_split=model_config.validation_split, callbacks=[early_stopping])  # Added early stopping
#             logger.info("Model training completed successfully.")

#             save_model(model, model_config.save_model_path)
#             logger.info("Model saved successfully!")

#             return model

#         except Exception as e:
#             logger.exception(f"Error occurred during model training: {e}")
#             raise e

# if __name__ == '__main__':
#     try:
#         logger.info(f">>>>>> stage {STAGE_NAME_MODEL_TRAINING} started <<<<<<")
#         obj = ModelTrainingPipeline()
#         obj.main()

#         logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_TRAINING} completed <<<<<<<")

#     except Exception as e:
#         logger.exception(e)
#         raise e


##RF classifier

# stage_03_model_training.py
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
            model_config = config.rf_get_model_training_config()

            # Initialize the ModelBuilder with config parameters
            model_builder = ModelBuilder(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                min_samples_split=model_config.min_samples_split,
                min_samples_leaf=model_config.min_samples_leaf,
                max_features=model_config.max_features,
                bootstrap=model_config.bootstrap,
                class_weight=model_config.class_weight,
                random_state=model_config.random_state,
                save_model_path=model_config.save_model_path
            )

            # Build the Random Forest model
            model = model_builder.build_model()

            # Train the model
            logger.info("Starting model training...")
            model.fit(X_train, Y_train)  # Random Forest does not need epochs or batch size
            logger.info("Model training completed successfully.")

            # Save the model
            save_model(model, model_config.save_model_path)
            logger.info("Model saved successfully!")

            return model

        except Exception as e:
            logger.exception(f"Error occurred during model training: {e}")
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME_MODEL_TRAINING} started <<<<<<")
        obj = ModelTrainingPipeline()

        # Make sure X_train and Y_train are available here
        # You can load them from preprocessed files or any other source
         # Replace with actual data loading function
        obj.main()

        logger.info(f">>>>>>> stage {STAGE_NAME_MODEL_TRAINING} completed <<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
