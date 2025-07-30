#!/usr/bin/env python3
"""
Setup script to prepare AutoGluon model for MLflow serving
"""

import os
import tarfile
import shutil
import mlflow
import mlflow.pyfunc
from autogluon.tabular import TabularPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_and_setup_model():
    """Extract AutoGluon model and create MLflow model structure"""
    
    model_dir = "/opt/ml/model"
    extracted_dir = "/opt/ml/model/extracted_model"
    mlflow_model_dir = "/opt/ml/model/mlflow_model"
    
    # Check if MLflow model is already set up
    if os.path.exists(mlflow_model_dir):
        logger.info("MLflow model already exists, skipping setup")
        return
    
    # Check if model is already extracted
    if not os.path.exists(extracted_dir):
        # Look for the model tar file
        model_file = None
        for filename in os.listdir(model_dir):
            if filename.endswith('.tar.gz') and 'model' in filename:
                model_file = os.path.join(model_dir, filename)
                break
        
        if not model_file:
            raise FileNotFoundError(f"No model tar.gz file found in {model_dir}. Files: {os.listdir(model_dir)}")
        
        logger.info(f"Extracting model from {model_file}")
        
        # Extract the model
        os.makedirs(extracted_dir, exist_ok=True)
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall(extracted_dir)
        
        # Remove the tar file to save space
        os.remove(model_file)
    
    # Load the model to get metadata
    logger.info(f"Loading AutoGluon model from {extracted_dir}")
    model = TabularPredictor.load(extracted_dir, require_py_version_match=False)
    
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
    feature_names = model.feature_metadata_in.get_features()
    sample_input = {name: 0.0 for name in feature_names}
    
    # Save the MLflow model
    mlflow.pyfunc.save_model(
        path=mlflow_model_dir,
        python_model=AutoGluonMLflowModel(),
        artifacts={"model": extracted_dir},
        conda_env=conda_env,
        code_paths=["/opt/ml/model/autogluon_model.py"],
        signature=mlflow.models.infer_signature(
            model_input=[sample_input],
            model_output=None  # Let MLflow infer from first prediction
        )
    )
    
    logger.info(f"MLflow model saved to {mlflow_model_dir}")
    logger.info(f"Model features: {feature_names}")
    logger.info(f"Problem type: {model.problem_type}")

if __name__ == "__main__":
    extract_and_setup_model()