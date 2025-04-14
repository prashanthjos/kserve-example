import os
import json
import joblib
import numpy as np
from typing import Dict, List

from kserve import Model, ModelServer, model_server
from kserve.storage import Storage

class FraudDetectionPredictor(Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.model = None
        self.threshold = 0.5
        self.ready = False

    def load(self):
        """Load the model from storage."""
        model_dir = Storage.download(self.model_dir)
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        
        # Load model configuration
        config_path = os.path.join(model_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.threshold = config.get('threshold', 0.5)
        
        self.ready = True
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """Make predictions on the input data."""
        instances = payload.get("instances", [])
        
        # Convert to numpy array
        inputs = np.asarray(instances)
        
        # Get model predictions
        predictions = self.model.predict(inputs).tolist()
        probabilities = self.model.predict_proba(inputs).tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }

if __name__ == "__main__":
    model_dir = os.environ.get('MODEL_DIR', 'pvc://kubeflow-artifact-storage/fraud-detection/v1')
    model = FraudDetectionPredictor("fraud-detection-predictor", model_dir)
    ModelServer(model).start(models=[model])