import numpy as np
import joblib
import json
from .model import FraudDetectionModel

class FraudDetectionPredictor:
    """Class for making predictions with the fraud detection model."""
    
    def __init__(self, model_path=None, model_instance=None, threshold=0.5):
        """Initialize the predictor with either a model path or a model instance."""
        self.model = None
        self.scaler = None
        self.threshold = threshold
        
        if model_instance is not None:
            self.model = model_instance
        elif model_path is not None:
            self.load(model_path)
        else:
            raise ValueError("Either model_path or model_instance must be provided")
    
    def load(self, model_path):
        """Load model and preprocessing components."""
        # Load the model
        self.model = FraudDetectionModel.load(model_path)
        
        # Load the scaler if available
        scaler_path = f"{model_path}/scaler.joblib"
        try:
            self.scaler = joblib.load(scaler_path)
        except:
            print("Scaler not found at {scaler_path}. Proceeding without scaling.")
        
        # Load threshold if available
        config_path = f"{model_path}/model_config.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.threshold = config.get('threshold', 0.5)
        except:
            pass
        
        return self
    
    def preprocess(self, inputs):
        """Preprocess input data before prediction."""
        # Convert to numpy array if it's not already
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        
        # Apply scaling if available
        if self.scaler is not None:
            inputs = self.scaler.transform(inputs)
        
        return inputs
    
    def predict(self, inputs, return_probabilities=False):
        """Make predictions on input data."""
        # Preprocess the inputs
        preprocessed_inputs = self.preprocess(inputs)
        
        # Get probabilities
        probabilities = self.model.predict_proba(preprocessed_inputs)
        
        # Apply threshold for binary classification
        predictions = (probabilities[:, 1] >= self.threshold).astype(int)
        
        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions
    
    def explain_prediction(self, input_data, feature_names=None):
        """Provide simple explanation for a prediction based on feature importance."""
        if not hasattr(self.model.model, 'feature_importances_'):
            return {"error": "Model does not support feature importance"}
        
        # Use model's feature names if available and none provided
        if feature_names is None and self.model.feature_names is not None:
            feature_names = self.model.feature_names
        
        # Default feature names if none available
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(input_data))]
        
        # Get feature importances
        importances = self.model.model.feature_importances_
        
        # Create a list of (feature_name, value, importance) tuples
        feature_data = []
        for i, (name, value) in enumerate(zip(feature_names, input_data)):
            feature_data.append({
                'name': name,
                'value': float(value),
                'importance': float(importances[i])
            })
        
        # Sort by importance
        feature_data.sort(key=lambda x: x['importance'], reverse=True)
        
        # Return the top features
        return {
            'top_features': feature_data[:5],
            'all_features': feature_data
        }