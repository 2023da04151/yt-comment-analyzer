import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging
import sys

# Logging setup
logger = logging.getLogger("model_registry")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_model_info(file_path):
    """Load model information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        return model_info
    except FileNotFoundError:
        logger.error(f"Model information file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        sys.exit(1)

def register_model(model_name : str, model_info : dict):
    """Register the model with DagsHub."""
    try:
        logger.info(f"Registering model: {model_name} with info: {model_info}")
        # Initialize MLflow client
        client = MlflowClient()
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        # Get the model version and name
        registered_model_version = model_version.version
        registered_model_name = model_version.name
        logger.info(f"Registered model version: {registered_model_version}, name: {registered_model_name}")
        # Set a tag on the model version to indicate deployment stage
        client.set_model_version_tag(
            name=registered_model_name,
            version=registered_model_version,
            key="deployment_stage",
            value="staging"
        )
        # Set an alias for the model version
        alias_name = "dissertation_project"
        client.set_registered_model_alias(registered_model_name, alias_name, registered_model_version)
        
        
        logger.info(f"Registering model: {model_name}")
        model_uri = f"models:/{model_name}/latest"

        logger.info(f"Model {model_name} registered successfully.")
    except Exception as e:
        logger.error(f"Failed to register model {model_name}: {e}")
        raise




def main():
    logger.info("Starting model registry...")
    
    try:
        # Initialize DagsHub
        logger.info("Initializing DagsHub...")
        mlflow.set_tracking_uri("https://dagshub.com/2023da04151/yt-comment-analyzer.mlflow")

        # Get project root
        project_root = Path.cwd()
        # model information file path
        run_info_path = project_root / "experiment_info.json"

        logger.info(f"Loading model information from: {run_info_path}")
        model_info = load_model_info(run_info_path)
        logger.info("Model information loaded successfully.")
        # Extract model name
        model_name  = "youtube_chrome_plugin_model"
        register_model(model_name, model_info)
        # Initialize MLflow client

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()