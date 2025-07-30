import bentoml
from autogluon.tabular import TabularPredictor
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.core.constants import REGRESSION
import pandas as pd
import os
import tarfile
from typing import Dict, Any, Union, List
from io import StringIO

# Model extraction and loading logic
def load_autogluon_model():
    """Extract and load the AutoGluon model"""
    model_dir = "/opt/ml/model"
    extracted_dir = "/opt/ml/model/extracted_model"
    
    # Check if model is already extracted
    if os.path.exists(extracted_dir):
        return TabularPredictor.load(extracted_dir, require_py_version_match=False)
    
    # Look for the model tar file
    model_file = None
    for filename in os.listdir(model_dir):
        if filename.endswith('.tar.gz') and 'model' in filename:
            model_file = os.path.join(model_dir, filename)
            break
    
    if not model_file:
        raise FileNotFoundError(f"No model tar.gz file found in {model_dir}. Files: {os.listdir(model_dir)}")
    
    # Extract the model
    os.makedirs(extracted_dir, exist_ok=True)
    with tarfile.open(model_file, 'r:gz') as tar:
        tar.extractall(extracted_dir)
    
    # Remove the tar file to save space
    os.remove(model_file)
    
    # Load the AutoGluon model
    return TabularPredictor.load(extracted_dir, require_py_version_match=False)

# Load the model once at startup
model = load_autogluon_model()

# Create BentoML service using the modern v1.4+ API
@bentoml.service(
    name="autogluon_predictor",
    resources={"cpu": "1000m", "memory": "2Gi"}
)
class AutoGluonService:
    
    @bentoml.api
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using AutoGluon model"""
        try:
            # Handle different input formats
            if "instances" in input_data:
                if isinstance(input_data["instances"], list):
                    data = pd.DataFrame(input_data["instances"])
                else:
                    data = pd.DataFrame([input_data["instances"]])
            else:
                data = pd.DataFrame([input_data])
            
            # Make predictions
            if model.problem_type != REGRESSION:
                pred_proba = model.predict_proba(data, as_pandas=True)
                pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
                pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
                pred.name = str(pred.name) + "_pred" if pred.name is not None else "pred"
                prediction = pd.concat([pred, pred_proba], axis=1)
            else:
                prediction = model.predict(data, as_pandas=True)
                
            if isinstance(prediction, pd.Series):
                prediction = prediction.to_frame()
            
            # Convert to JSON-serializable format
            result = prediction.to_dict(orient='records')
            
            return {
                "predictions": result,
                "model_info": {
                    "problem_type": str(model.problem_type),
                    "num_features": len(model.feature_metadata_in.get_features()),
                    "feature_names": model.feature_metadata_in.get_features()
                }
            }
            
        except Exception as e:
            return {"error": str(e), "predictions": []}

    @bentoml.api
    def predict_csv(self, csv_data: str) -> Dict[str, Any]:
        """Predict using CSV input data"""
        try:
            data = pd.read_csv(StringIO(csv_data))
            
            # Make predictions
            if model.problem_type != REGRESSION:
                pred_proba = model.predict_proba(data, as_pandas=True)
                pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
                pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
                pred.name = str(pred.name) + "_pred" if pred.name is not None else "pred"
                prediction = pd.concat([pred, pred_proba], axis=1)
            else:
                prediction = model.predict(data, as_pandas=True)
                
            if isinstance(prediction, pd.Series):
                prediction = prediction.to_frame()
            
            # Convert to JSON-serializable format
            result = prediction.to_dict(orient='records')
            
            return {
                "predictions": result,
                "model_info": {
                    "problem_type": str(model.problem_type),
                    "num_features": len(model.feature_metadata_in.get_features()),
                    "feature_names": model.feature_metadata_in.get_features()
                }
            }
            
        except Exception as e:
            return {"error": str(e), "predictions": []}

    @bentoml.api
    def health(self) -> Dict[str, Union[str, bool]]:
        """Health check endpoint"""
        return {"status": "healthy", "model_loaded": True}

    @bentoml.api
    def model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "problem_type": str(model.problem_type),
            "num_features": len(model.feature_metadata_in.get_features()),
            "feature_names": model.feature_metadata_in.get_features(),
            "model_path": "/opt/ml/model/extracted_model"
        }