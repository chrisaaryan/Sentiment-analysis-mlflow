from sentimentAnalysis.constants import *
from sentimentAnalysis.utils.common import read_yaml, create_directories
from sentimentAnalysis.entity.config_entity import DataIngestionConfig
# from sentimentAnalysis.entity.config_entity import DataValidationConfig
from sentimentAnalysis.entity.config_entity import DataPreprocessingConfig
from sentimentAnalysis.entity.config_entity import ModelTrainingConfig
from sentimentAnalysis.entity.config_entity import EvaluationConfig
from sentimentAnalysis.entity.config_entity import RFModelTrainingConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath= CONFIG_FILE_PATH,
            params_filepath= PARAMS_FILE_PATH):
        self.config= read_yaml(config_filepath)
        self.params= read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config= self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config= DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )

        return data_ingestion_config
    
    # def get_data_validation_config(self):
    #     config = self.config['data_validation']
    #     return DataValidationConfig(
    #         expected_columns=config['expected_columns'],
    #         sentiment_values=config['sentiment_values']
    #     )
    
    def get_data_preprocessing_config(self)->DataPreprocessingConfig:
        config = self.config['data_preprocessing']
        return DataPreprocessingConfig(
            max_words=config['max_words'],
            max_sequence_length=config['max_sequence_length'],
            train_test_split_ratio=config['train_test_split_ratio'],
            random_state=config['random_state'],
            train_data_path=config['train_data_path'],
            test_data_path=config['test_data_path']
        )
    
    def get_model_training_config(self):
    # Accessing the model configuration from params
        model_config = self.params['model_config']
        return ModelTrainingConfig(
            input_dim=model_config['input_dim'],
            output_dim=model_config['output_dim'],
            input_length=model_config['input_length'],
            lstm_units=model_config['lstm_units'],
            dropout_rate=model_config['dropout_rate'],
            recurrent_dropout_rate=model_config['recurrent_dropout_rate'],
            optimizer=model_config['optimizer'],
            learning_rate=model_config['learning_rate'],
            loss=model_config['loss'],
            metrics=model_config['metrics'],
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            validation_split=model_config['validation_split'],
            early_stopping_monitor=model_config['early_stopping']['monitor'], 
            early_stopping_patience=model_config['early_stopping']['patience'], 
            save_model_path=model_config['save_model_path']
    )

    def rf_get_model_training_config(self):
    # Accessing the model configuration from params
        model_config = self.params['model_config']
        return RFModelTrainingConfig(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            min_samples_split=model_config['min_samples_split'],
            min_samples_leaf=model_config['min_samples_leaf'],
            max_features=model_config['max_features'],
            bootstrap=model_config['bootstrap'],
            class_weight=model_config['class_weight'],
            random_state=model_config['random_state'],
            save_model_path=model_config['save_model_path']
    )


    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=Path(self.params['model_config']['save_model_path']),
            # test_data=Path(self.config['data_preprocessing']['test_data_path']),
            mlflow_uri=self.config['mlflow_config']['mlflow_uri'],
            all_params=self.params
        )
        return eval_config