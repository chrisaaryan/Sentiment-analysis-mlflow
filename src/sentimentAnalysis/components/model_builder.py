# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from sentimentAnalysis import logger

# class ModelBuilder:
#     def __init__(self, input_dim, output_dim, input_length, lstm_units, dropout_rate, recurrent_dropout_rate, optimizer, learning_rate, loss, metrics, epochs, batch_size, validation_split, early_stopping_monitor, early_stopping_patience, save_model_path):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_length = input_length
#         self.lstm_units = lstm_units
#         self.dropout_rate = dropout_rate
#         self.recurrent_dropout_rate = recurrent_dropout_rate  
#         self.optimizer = optimizer
#         self.learning_rate = learning_rate  
#         self.loss = loss
#         self.metrics = metrics
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.validation_split = validation_split
#         self.early_stopping_monitor = early_stopping_monitor  
#         self.early_stopping_patience = early_stopping_patience 
#         self.save_model_path = save_model_path

#     def build_model(self):
#         """
#         Build and compile the LSTM model.
#         """
#         try:
#             logger.info("Building the model...")

#             # Define the Sequential model
#             model = Sequential()

#             # Add layers
#             model.add(Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length))
#             model.add(LSTM(self.lstm_units, dropout=self.dropout_rate, recurrent_dropout=self.recurrent_dropout_rate))  # Added recurrent dropout
#             model.add(Dense(1, activation="sigmoid"))

#             # Compile the model with custom learning rate
#             if self.optimizer == 'adam':
#                 optimizer = Adam(learning_rate=self.learning_rate)  # Custom learning rate for Adam
#             else:
#                 optimizer = self.optimizer

#             model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

#             logger.info("Model built and compiled successfully.")
#             model.summary()

#             return model

#         except Exception as e:
#             logger.exception(f"Error occurred while building the model: {e}")
#             raise e

#     def get_early_stopping_callback(self):
#         """
#         Create and return the early stopping callback.
#         """
#         return EarlyStopping(
#             monitor=self.early_stopping_monitor,
#             patience=self.early_stopping_patience,
#             mode='min',
#             verbose=1
#         )

##RF Classifier
# model_builder.py
from sklearn.ensemble import RandomForestClassifier

class ModelBuilder:
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                 max_features, bootstrap, class_weight, random_state, save_model_path):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        self.save_model_path = save_model_path

    def build_model(self):
        # Build and return the Random Forest Classifier
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
