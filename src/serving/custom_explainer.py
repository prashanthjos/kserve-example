import os
import json
import joblib
import numpy as np
import shap
from typing import Dict, List

from kserve import Model, ModelServer
from kserve.storage import Storage

class FraudDetectionExplainer(Model):
    def __init__(self, name: str, model_dir: str, predictor_host: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.predictor_host = predictor_host
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.ready = False

    def load(self):
        """Load the model and create explainer."""
        model_dir = Storage.download(self.model_dir)
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        
        # Load feature names
        with open(os.path.join(model_dir, "feature_names.json"), 'r') as f:
            self.feature_names = json.load(f)
        
        # Create the explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.ready = True
        return self.ready

    def explain(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """Generate explanations for the input data."""
        instances = np.asarray(payload.get("instances", []))
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(instances)
        
        # If we have a binary classification model
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # For binary classification we focus on class 1 (fraud)
            shap_values = shap_values[1]
        
        # Format the response
        explanations = []
        for i, instance in enumerate(instances):
            # Get the SHAP values for this instance
            instance_shap = shap_values[i].tolist()
            
            # Create the feature importance map
            feature_importance = {}
            for j, feature in enumerate(self.feature_names):
                feature_importance[feature] = instance_shap[j]
            
            # Get the base value (expected value) from the explainer
            base_value = float(self.explainer.expected_value) 
            if isinstance(self.explainer.expected_value, list):
                base_value = float(self.explainer.expected_value[1])
                
            explanations.append({
                "base_value": base_value,
                "feature_importance": feature_importance,
                "shap_values": instance_shap
            })
        
        return {"explanations": explanations}

if __name__ == "__main__":
    model_dir = os.environ.get('MODEL_DIR', 'pvc://kubeflow-artifact-storage/fraud-detection/v1')
    predictor_host = os.environ.get('PREDICTOR_HOST', 'localhost:8080')
    model = FraudDetectionExplainer("fraud-detection-explainer", model_dir, predictor_host)
    ModelServer(model).start(models=[model])