import mlflow
# import mlflow.keras
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
from sentimentAnalysis.utils.common import save_json
from sentimentAnalysis.config.configuration import EvaluationConfig
from sentimentAnalysis import logger
import os

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    # LSTM
    # @staticmethod
    # def load_model(path: Path) -> tf.keras.Model:
    #     return tf.keras.models.load_model(path)

    #RFClassifier
    @staticmethod
    def load_model(path: Path):
        """Load RandomForestClassifier model."""
        return joblib.load(path)
    
    # def _load_test_data(self):
    #     # Load test data (assuming 'preprocessed_review' is already tokenized and padded)
    #     df = pd.read_csv(self.config.test_data)
    #     X_test = df['preprocessed_review'].values  # Use the preprocessed text
    #     Y_test = df['sentiment'].values  # Labels (either binary or categorical)
    #     return X_test, Y_test
    
    # LSTM
    # def _evaluate_model(self, X_test, Y_test):
    #     # Use the tokenizer saved during training if applicable
    #     tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.config.all_params['model_config']['input_dim'])
    #     # Assume tokenizer has been fit on training data
    #     X_test_seq = tokenizer.texts_to_sequences(X_test)  # Tokenize test data
    #     X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=self.config.all_params['model_config']['input_length'])

    #     # Perform predictions
    #     predictions = self.model.predict(X_test_pad)
    #     predictions = (predictions > 0.5).astype(int)  # Assuming binary classification

    #     # Calculate evaluation metrics
    #     accuracy = accuracy_score(Y_test, predictions)
    #     precision = precision_score(Y_test, predictions)
    #     recall = recall_score(Y_test, predictions)
    #     f1 = f1_score(Y_test, predictions)

    #     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    # RFClassifier
    def _evaluate_model(self, X_test, Y_test):
        # Perform predictions using the loaded RandomForestClassifier
        predictions = self.model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(Y_test, predictions)
        precision = precision_score(Y_test, predictions, average='weighted')
        recall = recall_score(Y_test, predictions, average='weighted')
        f1 = f1_score(Y_test, predictions, average='weighted')

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    def evaluation(self, X_test, Y_test):
        self.model = self.load_model(self.config.path_of_model)
        # X_test, Y_test = self._load_test_data()
        self.metrics = self._evaluate_model(X_test, Y_test)
        self.save_score()

    def save_score(self):
        # Save metrics as JSON
        save_json(path=Path("evaluation_scores.json"), data=self.metrics)

    def log_into_mlflow(self):
        os.environ['MLFLOW_TRACKING_URI']= 'https://dagshub.com/chrisaaryan/my-first-repo.mlflow'
        os.environ['MLFLOW_TRACKING_USERNAME']= 'chrisaaryan'
        os.environ['MLFLOW_TRACKING_PASSWORD']= '97f85b6ca1fc224edc2875d929cf45a081ca85b1'
        # Set MLflow tracking URI
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Start logging with MLflow
        with mlflow.start_run():
            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.metrics)

            # Log the model to MLflow
            if tracking_url_type_store != "file":
                print("model logged")
                mlflow.sklearn.log_model(self.model, "model", registered_model_name="SentimentAnalysisModel2")
            else:
                print("model logged")
                mlflow.sklearn.log_model(self.model, "model")

    # def log_into_mlflow(self):
    #     try:
    #         print("entered")
    #         mlflow.set_tracking_uri(self.config.mlflow_uri)  # DAGsHub tracking URI
    #         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    #         print("moved")

    #         # Log into MLflow
    #         with mlflow.start_run():
    #             print("entered_again")

    #             # Log parameters and metrics
    #             mlflow.log_params(self.config.all_params)
    #             mlflow.log_metrics(self.metrics)

    #             if tracking_url_type_store != "file":
    #                 print("model logged")
    #                 mlflow.sklearn.log_model(self.model, "model", registered_model_name="SentimentAnalysisModel2")
    #             else:
    #                 print("model logged")
    #                 mlflow.sklearn.log_model(self.model, "model")
    #         print(f"🏃 View run at: {mlflow.get_run(mlflow.active_run().info.run_id).info.artifact_uri}")
    #         print(f"🧪 View experiment at: {mlflow.get_tracking_uri()}")

    #     except Exception as e:
    #         logger.exception(f"Error during MLflow logging: {e}")
    #         raise e