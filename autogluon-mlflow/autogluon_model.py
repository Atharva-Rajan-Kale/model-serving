import mlflow
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.core.constants import REGRESSION
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoGluonMLflowModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for AutoGluon TabularPredictor"""
    
    def load_context(self, context):
        """Load the AutoGluon model"""
        model_path = context.artifacts["model"]
        logger.info(f"Loading AutoGluon model from {model_path}")
        self.model = TabularPredictor.load(model_path, require_py_version_match=False)
        logger.info(f"Model loaded successfully. Problem type: {self.model.problem_type}")
        logger.info(f"Features: {self.model.feature_metadata_in.get_features()}")
    
    def predict(self, context, model_input):
        """Make predictions using the AutoGluon model"""
        try:
            # Convert input to DataFrame if it's not already
            if not isinstance(model_input, pd.DataFrame):
                if isinstance(model_input, dict):
                    model_input = pd.DataFrame([model_input])
                elif isinstance(model_input, list):
                    model_input = pd.DataFrame(model_input)
                else:
                    model_input = pd.DataFrame(model_input)
            
            # Make predictions based on problem type
            if self.model.problem_type != REGRESSION:
                # Classification: return both predictions and probabilities
                pred_proba = self.model.predict_proba(model_input, as_pandas=True)
                pred = get_pred_from_proba_df(pred_proba, problem_type=self.model.problem_type)
                
                # Combine predictions and probabilities
                pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
                pred.name = str(pred.name) + "_pred" if pred.name is not None else "pred"
                result = pd.concat([pred, pred_proba], axis=1)
            else:
                # Regression: return predictions only
                result = self.model.predict(model_input, as_pandas=True)
                if isinstance(result, pd.Series):
                    result = result.to_frame()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e