from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sentimentAnalysis import logger

class ModelBuilder:
    def __init__(self, input_dim, output_dim, input_length, lstm_units, dropout_rate, optimizer, loss, metrics, epochs, batch_size, validation_split):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        try:
            logger.info("Building the model...")

            # Define the Sequential model
            model = Sequential()

            # Add layers
            model.add(Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length))
            model.add(LSTM(self.lstm_units, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
            model.add(Dense(1, activation="sigmoid"))

            # Compile the model
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

            logger.info("Model built and compiled successfully.")
            model.summary()

            return model

        except Exception as e:
            logger.exception(f"Error occurred while building the model: {e}")
            raise e