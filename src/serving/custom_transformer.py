import json
import numpy as np
import joblib
from typing import Dict, List

from kserve import KServeClient
from kserve.model import ModelServer, Model
from kserve.storage import Storage
from kserve.utils.utils import get_predict_input, get_predict_response
from kserve.protocol.rest.server import RESTServer
from kserve.protocol.infer_type import InferRequest, InferResponse

class FraudDetectionTransformer(Model):
    def __init__(self, name: str, predictor_host: str, model_dir: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.model_dir = model_dir
        self.ready = False
        self.scaler = None
        self.feature_names = None

    def load(self):
        model_dir = Storage.download(self.model_dir)
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        
        with open(f"{model_dir}/feature_names.json", "r") as f:
            self.feature_names = json.load(f)
            
        self.ready = True
        return self.ready

    def preprocess(self, inputs: Dict) -> Dict:
        """Preprocess the inputs for model prediction."""
        instances = inputs.get("instances", [])
        processed_inputs = []
        
        for instance in instances:
            # Check if the input is in the correct format
            if isinstance(instance, dict):
                # Extract features in the correct order
                features = [instance.get(feature, 0) for feature in self.feature_names]
                processed_inputs.append(features)
            elif isinstance(instance, list):
                # If already a list, use as is
                processed_inputs.append(instance)
        
        # Apply scaling
        scaled_inputs = self.scaler.transform(np.array(processed_inputs))
        
        # Return in the format expected by the predictor
        return {"instances": scaled_inputs.tolist()}

    def postprocess(self, outputs: Dict) -> Dict:
        """Postprocess the model predictions."""
        predictions = outputs.get("predictions", [])
        
        # Add confidence scores if available
        if "probabilities" in outputs:
            probabilities = outputs.get("probabilities", [])
            processed_outputs = []
            
            for i, pred in enumerate(predictions):
                result = {
                    "prediction": int(pred),
                    "confidence": float(probabilities[i][int(pred)])
                }
                processed_outputs.append(result)
        else:
            # Just return predictions if no probabilities
            processed_outputs = [{"prediction": int(p)} for p in predictions]
            
        return {"predictions": processed_outputs}

if __name__ == "__main__":
    model = FraudDetectionTransformer(
        name="fraud-detection-transformer", 
        predictor_host="localhost:8080",
        model_dir="pvc://kubeflow-artifact-storage/fraud-detection/v1"
    )
    ModelServer(model).start(models=[model])