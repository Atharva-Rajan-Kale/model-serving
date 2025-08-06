import bentoml
from autogluon.tabular import TabularPredictor
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.core.constants import REGRESSION
import pandas as pd
import os
import tarfile
from typing import Dict, Any, Union, List
from io import StringIO
import json

# Model extraction and loading logic
def load_autogluon_model():
    """Extract and load the AutoGluon model"""
    model_dir = os.environ.get("MODEL_PATH", "/opt/ml/model")
    extracted_dir = os.path.join(model_dir, "extracted_model")
    
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
    
    # Load the AutoGluon model
    return TabularPredictor.load(extracted_dir, require_py_version_match=False)

# Create BentoML service
@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 300}
)
class AutoGluonService:
    
    def __init__(self):
        self.model = None
    
    def _get_model(self):
        """Lazy load the model when first needed"""
        if self.model is None:
            self.model = load_autogluon_model()
        return self.model
    
    def _predict_logic(self, input_data):
        """Core prediction logic"""
        try:
            # Handle CSV string input
            if isinstance(input_data, str):
                data = pd.read_csv(StringIO(input_data))
            # Handle different JSON input formats
            elif isinstance(input_data, list):
                # Direct list of records (batch format)
                data = pd.DataFrame(input_data)
            elif "instances" in input_data:
                # Wrapped in instances key
                if isinstance(input_data["instances"], list):
                    data = pd.DataFrame(input_data["instances"])
                else:
                    data = pd.DataFrame([input_data["instances"]])
            elif isinstance(input_data, dict) and len(input_data) > 0:
                # Check if it's a single record or contains batch data
                first_value = next(iter(input_data.values()))
                if isinstance(first_value, list) and len(first_value) > 0 and isinstance(first_value[0], (int, float, str)):
                    # This looks like batch data where each key has a list of values
                    data = pd.DataFrame(input_data)
                else:
                    # Single record
                    data = pd.DataFrame([input_data])
            else:
                data = pd.DataFrame([input_data])
            
            # Get the model (lazy loading)
            model = self._get_model()
            
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
                    "feature_names": model.feature_metadata_in.get_features(),
                    "batch_size": len(data)
                }
            }
            
        except Exception as e:
            return {"error": str(e), "predictions": []}
    
    @bentoml.api
    def predict(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], str]) -> Dict[str, Any]:
        """Standard BentoML prediction endpoint"""
        return self._predict_logic(input_data)

    @bentoml.api
    def health(self) -> Dict[str, Union[str, bool]]:
        """Health check endpoint"""
        return {"status": "healthy"}

    @bentoml.api
    def model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            model = self._get_model()
            return {
                "problem_type": str(model.problem_type),
                "num_features": len(model.feature_metadata_in.get_features()),
                "feature_names": model.feature_metadata_in.get_features(),
                "model_path": os.environ.get("MODEL_PATH", "/opt/ml/model")
            }
        except Exception as e:
            return {"error": str(e), "model_loaded": False}

    # Additional endpoints for SageMaker compatibility
    @bentoml.api
    def ping(self) -> Dict[str, str]:
        """SageMaker health check endpoint (accessible at /ping)"""
        return {"status": "healthy"}

    @bentoml.api  
    def invocations(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], str]) -> Dict[str, Any]:
        """SageMaker prediction endpoint (accessible at /invocations)"""
        return self._predict_logic(input_data)