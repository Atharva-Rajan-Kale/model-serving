#!/usr/bin/env python3
"""
Setup script to prepare AutoGluon model for MLflow serving
"""

import os
import tarfile
import shutil
import mlflow
import mlflow.pyfunc
import pandas as pd
from autogluon.tabular import TabularPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_and_setup_model():
    """Extract AutoGluon model and create MLflow model structure"""
    
    # Ensure imports are available (defensive programming)
    import shutil
    import os
    
    model_dir = "/opt/ml/model"
    extracted_dir = "/opt/ml/model/extracted_model"
    mlflow_model_dir = "/opt/ml/model/mlflow_model"
    
    # Check if MLflow model is already set up and valid
    if os.path.exists(mlflow_model_dir):
        mlmodel_file = os.path.join(mlflow_model_dir, "MLmodel")
        if os.path.exists(mlmodel_file):
            logger.info("Valid MLflow model already exists, skipping setup")
            return
        else:
            logger.info("MLflow model directory exists but is incomplete, attempting cleanup...")
            try:
                # Try to remove the directory
                shutil.rmtree(mlflow_model_dir)
                logger.info("Cleanup successful")
            except (OSError, PermissionError) as e:
                logger.warning(f"Cannot cleanup directory (read-only filesystem?): {e}")
                logger.info("Continuing with setup in a different location...")
                # Use a different directory name to avoid conflicts
                import time
                mlflow_model_dir = f"/opt/ml/model/mlflow_model_{int(time.time())}"
    
    # Check if model is already extracted
    if not os.path.exists(extracted_dir):
        # Look for the model tar file in various locations
        model_file = None
        
        # First, check for direct tar.gz files
        for filename in os.listdir(model_dir):
            if filename.endswith('.tar.gz') and 'model' in filename:
                model_file = os.path.join(model_dir, filename)
                break
        
        # If not found, check inside subdirectories (nested structure)
        if not model_file:
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Look inside subdirectories
                    try:
                        for subfile in os.listdir(item_path):
                            if subfile.endswith('.tar.gz') and 'model' in subfile:
                                model_file = os.path.join(item_path, subfile)
                                logger.info(f"Found model file in subdirectory: {model_file}")
                                break
                        if model_file:
                            break
                    except (OSError, PermissionError):
                        continue
        
        if not model_file:
            logger.error(f"No model tar.gz file found in {model_dir}")
            logger.error(f"Directory contents: {os.listdir(model_dir)}")
            # Try to list subdirectory contents for debugging
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    try:
                        logger.error(f"Contents of {item}: {os.listdir(item_path)}")
                    except:
                        pass
            raise FileNotFoundError(f"No model tar.gz file found in {model_dir}. Files: {os.listdir(model_dir)}")
        
        logger.info(f"Extracting model from {model_file}")
        
        # Extract the model
        os.makedirs(extracted_dir, exist_ok=True)
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall(extracted_dir)
        
        # Try to remove the tar file to save space (skip if read-only)
        try:
            os.remove(model_file)
            logger.info("Removed original tar file to save space")
        except (OSError, PermissionError):
            logger.info("Cannot remove tar file (read-only filesystem) - continuing")
    
    # Load the model to get metadata
    logger.info(f"Loading AutoGluon model from {extracted_dir}")
    
    # Find the actual model directory (might be nested)
    model_path = extracted_dir
    if os.path.exists(os.path.join(extracted_dir, "mlflow_model")):
        # If there's already an mlflow_model directory, look for the actual model
        for item in os.listdir(extracted_dir):
            item_path = os.path.join(extracted_dir, item)
            if os.path.isdir(item_path) and item != "mlflow_model":
                # Check if this looks like an AutoGluon model directory
                if any(f.endswith('.pkl') or f == 'predictor.pkl' for f in os.listdir(item_path)):
                    model_path = item_path
                    break
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Contents: {os.listdir(model_path)}")
    
    try:
        model = TabularPredictor.load(model_path, require_py_version_match=False)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        # Try to find the model in subdirectories
        for root, dirs, files in os.walk(extracted_dir):
            if 'predictor.pkl' in files or any(f.endswith('.pkl') for f in files):
                logger.info(f"Found potential model directory: {root}")
                try:
                    model = TabularPredictor.load(root, require_py_version_match=False)
                    model_path = root
                    break
                except:
                    continue
        else:
            raise e
    
    # Create MLflow model
    logger.info("Creating MLflow model structure")
    
    # Copy the autogluon_model.py to the model directory so MLflow can find it
    import shutil
    shutil.copy("/opt/ml/autogluon_model.py", "/opt/ml/model/autogluon_model.py")
    
    # Import the custom model class
    import sys
    sys.path.append('/opt/ml/model')
    from autogluon_model import AutoGluonMLflowModel
    
    # Create conda environment specification
    conda_env = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=3.11.9',
            'pip',
            {
                'pip': [
                    'mlflow',
                    f'autogluon=={model._trainer.version}' if hasattr(model._trainer, 'version') else 'autogluon>=1.3.1',
                    'pandas',
                    'numpy',
                    'scikit-learn'
                ]
            }
        ],
        'name': 'autogluon_env'
    }
    
    # Create sample input for signature inference
    try:
        feature_names = model.feature_metadata_in.get_features()
        logger.info(f"Model features: {feature_names}")
        
        # Create a more realistic sample input based on feature types
        sample_input = {}
        for feature in feature_names:
            # Use a simple default value - MLflow will handle type inference
            sample_input[feature] = 0
        
        sample_df = pd.DataFrame([sample_input])
        logger.info(f"Sample input created: {sample_df.dtypes}")
        
        # Save the MLflow model without signature first (to avoid inference issues)
        logger.info("Saving MLflow model...")
        mlflow.pyfunc.save_model(
            path=mlflow_model_dir,
            python_model=AutoGluonMLflowModel(),
            artifacts={"model": model_path},
            conda_env=conda_env,
            code_paths=["/opt/ml/model/autogluon_model.py"]
            # Skip signature for now to avoid issues
        )
        
    except Exception as e:
        logger.error(f"Error during model saving: {e}")
        # Try saving without signature as fallback
        logger.info("Attempting fallback model save without signature...")
        mlflow.pyfunc.save_model(
            path=mlflow_model_dir,
            python_model=AutoGluonMLflowModel(),
            artifacts={"model": model_path},
            conda_env=conda_env,
            code_paths=["/opt/ml/model/autogluon_model.py"]
        )
    
    logger.info(f"MLflow model saved to {mlflow_model_dir}")
    logger.info(f"Model features: {feature_names}")
    logger.info(f"Problem type: {model.problem_type}")
    
    # Validate the MLflow model was created correctly
    mlmodel_file = os.path.join(mlflow_model_dir, "MLmodel")
    if os.path.exists(mlmodel_file):
        logger.info("✅ MLflow model validation successful - MLmodel file exists")
        with open(mlmodel_file, 'r') as f:
            logger.info(f"MLmodel contents preview: {f.read()[:200]}...")
    else:
        logger.error("❌ MLflow model validation failed - MLmodel file missing")
        raise FileNotFoundError(f"MLmodel file not created at {mlmodel_file}")

if __name__ == "__main__":
    extract_and_setup_model()