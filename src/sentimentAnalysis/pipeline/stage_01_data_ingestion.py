from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.data_ingestion import DataIngestion
from sentimentAnalysis import logger
import pandas  as pd
import os

STAGE_NAME= "Data Ingestion Stage"
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self) -> pd.DataFrame:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # Check if data is already extracted
        # extracted_file_path = os.path.join(data_ingestion_config.unzip_dir, "IMDB Dataset.csv")
        # if os.path.exists(extracted_file_path):
        # logger.info("Ingested data already exists, skipping download and extraction.")
        # data = pd.read_csv(extracted_file_path)
        # else:
            # Download and extract if not present
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data = data_ingestion.read_data()
        output_dir = 'artifacts/data_ingestion'
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame directly to CSV
        output_file_path = os.path.join(output_dir, "IMDB_Dataset.csv")
        logger.info(f"Saving ingested data to {output_file_path}")
        pd.DataFrame(data).to_csv(output_file_path, index=False)
        return data

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj= DataIngestionTrainingPipeline()
        data= obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e