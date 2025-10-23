"""
Advanced Custom Transformer for Fraud Detection
Features:
- Input data validation and error handling
- Advanced feature engineering
- Multiple preprocessing strategies
- Performance monitoring
- Detailed logging
"""


import json
import os
import numpy as np
import joblib
import time
import argparse
from typing import Dict, List, Union, Optional
from datetime import datetime

from kserve import Model, ModelServer, model_server, logging, InferRequest, InferResponse
from kserve.model import PredictorProtocol, PredictorConfig
from kserve.storage import Storage
from kserve.errors import InvalidInput

# Configure logging
logging.configure_logging()


class AdvancedFraudTransformer(Model):
    """
    Advanced transformer with comprehensive preprocessing capabilities.
    
    Features:
    - Data validation and cleaning
    - Feature scaling and normalization
    - Feature engineering (aggregations, ratios, etc.)
    - Error handling and logging
    - Performance metrics
    """
    
    def __init__(self, name: str, model_dir: str, predictor_config: PredictorConfig):
        super().__init__(name, predictor_config, return_response_headers=True)
        self.model_dir = model_dir
        self.ready = False
        
        # Model artifacts
        self.scaler = None
        self.feature_names = None
        self.feature_stats = None
        
        # Validation thresholds
        self.validation_enabled = True
        self.outlier_threshold = 5.0  # Z-score threshold
        
        # Monitoring
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0

    def load(self):
        """Load model artifacts and configurations."""
        try:
            model_dir = Storage.download(self.model_dir)
            
            if not os.path.exists(model_dir):
                self.ready = False
                return False
            
            # Load scaler
            scaler_path = f"{model_dir}/scaler.joblib"
            if not os.path.exists(scaler_path):
                self.ready = False
                return False
            
            self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            feature_names_path = f"{model_dir}/feature_names.json"
            if not os.path.exists(feature_names_path):
                self.ready = False
                return False
            
            with open(feature_names_path, "r") as f:
                self.feature_names = json.load(f)
            
            # Load feature statistics if available
            try:
                feature_stats_path = f"{model_dir}/feature_stats.json"
                if os.path.exists(feature_stats_path):
                    with open(feature_stats_path, "r") as f:
                        self.feature_stats = json.load(f)
                else:
                    self.feature_stats = None
            except Exception as e:
                self.feature_stats = None
            
            self.ready = True
            return True
            
        except Exception as e:
            self.ready = False
            return False

    def validate_input(self, instance: Union[Dict, List]) -> tuple[bool, str]:
        """
        Validate input data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if input is dict or list
            if isinstance(instance, dict):
                # Validate all required features are present
                missing_features = [f for f in self.feature_names if f not in instance]
                if missing_features:
                    return False, f"Missing features: {', '.join(missing_features[:5])}"
                
                # Check for null values
                null_features = [k for k, v in instance.items() if v is None]
                if null_features:
                    return False, f"Null values in features: {', '.join(null_features[:5])}"
                
                # Validate data types
                for feature in self.feature_names:
                    value = instance.get(feature)
                    if not isinstance(value, (int, float)):
                        return False, f"Invalid type for feature {feature}: expected number, got {type(value)}"
                
            elif isinstance(instance, list):
                # Validate length
                if len(instance) != len(self.feature_names):
                    return False, f"Expected {len(self.feature_names)} features, got {len(instance)}"
                
                # Validate all are numbers
                if not all(isinstance(x, (int, float)) for x in instance):
                    return False, "All features must be numbers"
                
                # Check for None values
                if any(x is None for x in instance):
                    return False, "Features cannot contain null values"
            else:
                return False, f"Invalid input type: expected dict or list, got {type(instance)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def detect_outliers(self, features: np.ndarray) -> List[int]:
        """
        Detect outliers using z-score method.
        
        Args:
            features: Array of feature values
            
        Returns:
            List of feature indices that are outliers
        """
        if self.feature_stats is None:
            return []
        
        outliers = []
        for i, (value, feature_name) in enumerate(zip(features, self.feature_names)):
            if feature_name in self.feature_stats:
                mean = self.feature_stats[feature_name].get('mean', 0)
                std = self.feature_stats[feature_name].get('std', 1)
                
                if std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > self.outlier_threshold:
                        outliers.append(i)
        
        return outliers

    def engineer_features(self, features: np.ndarray) -> Dict[str, float]:
        """
        Create engineered features from raw inputs.
        
        Args:
            features: Original feature array
            
        Returns:
            Dictionary of engineered features
        """
        engineered = {}
        
        # Statistical aggregations
        engineered['mean'] = float(np.mean(features))
        engineered['std'] = float(np.std(features))
        engineered['min'] = float(np.min(features))
        engineered['max'] = float(np.max(features))
        engineered['range'] = engineered['max'] - engineered['min']
        
        # Distribution features
        engineered['skewness'] = float(self._calculate_skewness(features))
        engineered['kurtosis'] = float(self._calculate_kurtosis(features))
        
        # Count features
        engineered['num_zeros'] = int(np.sum(features == 0))
        engineered['num_negative'] = int(np.sum(features < 0))
        
        return engineered

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Preprocess the inputs with validation and feature engineering.
        
        Args:
            inputs: Input dictionary with 'instances' key or InferRequest object
            headers: Optional request headers
            
        Returns:
            Processed inputs ready for model prediction
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            
            # Handle KServe v2 InferRequest object
            if hasattr(inputs, 'inputs'):
                instances = []
                for inp in inputs.inputs:
                    instances.extend(inp.data)
            # Handle dictionary format
            elif isinstance(inputs, dict):
                instances = inputs.get("instances", [])
            else:
                instances = inputs if isinstance(inputs, list) else []
            
            if not instances:
                raise InvalidInput("No instances provided in request")
            
            processed_inputs = []
            metadata_list = []
            
            for idx, instance in enumerate(instances):
                # Validate input
                is_valid, error_msg = self.validate_input(instance)
                if not is_valid:
                    if self.validation_enabled:
                        raise InvalidInput(f"Instance {idx}: {error_msg}")
                
                # Extract features
                if isinstance(instance, dict):
                    features = np.array([instance.get(feature, 0) for feature in self.feature_names])
                else:
                    features = np.array(instance)
                
                # Detect outliers
                outlier_indices = self.detect_outliers(features)
                
                # Engineer additional features
                engineered_features = self.engineer_features(features)
                
                # Store metadata
                metadata = {
                    'instance_id': idx,
                    'timestamp': datetime.utcnow().isoformat(),
                    'has_outliers': len(outlier_indices) > 0,
                    'outlier_features': [self.feature_names[i] for i in outlier_indices],
                    'engineered_features': engineered_features
                }
                metadata_list.append(metadata)
                
                processed_inputs.append(features)
            
            # Apply scaling
            scaled_inputs = self.scaler.transform(np.array(processed_inputs))
            
            # Calculate latency
            latency = time.time() - start_time
            self.total_latency += latency
            
            
            # Return in KServe v2 format if input was InferRequest
            if hasattr(inputs, 'inputs'):
                from kserve.protocol.infer_type import InferInput
                # Modify the input request with processed data
                # Create new InferInput with the scaled data
                infer_input = InferInput(
                    name="input-0",
                    shape=list(scaled_inputs.shape),
                    datatype="FP32",
                    data=scaled_inputs.flatten().tolist()
                )
                # Replace the inputs in the original request
                inputs.inputs = [infer_input]
                return inputs
            else:
                # Return with metadata for v1 protocol
                return {
                    "instances": scaled_inputs.tolist(),
                    "metadata": metadata_list
                }
            
        except InvalidInput:
            self.error_count += 1
            raise
        except Exception as e:
            self.error_count += 1
            raise InvalidInput(f"Preprocessing failed: {str(e)}")

    def postprocess(
        self,
        infer_response: Union[Dict, InferResponse],
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Union[Dict, InferResponse]:
        """
        Postprocess model predictions with enhanced formatting.
        
        Args:
            infer_response: Raw model response (Dict or InferResponse)
            headers: Optional request headers
            response_headers: Optional response headers
            
        Returns:
            Formatted predictions with metadata
        """
        try:
            # Handle InferResponse object
            if isinstance(infer_response, InferResponse):
                # For v2 protocol, return the response as-is or process as needed
                return infer_response
            
            # Handle dictionary format
            predictions = infer_response.get("predictions", [])
            probabilities = infer_response.get("probabilities", [])
            metadata_list = infer_response.get("metadata", [])
            
            processed_outputs = []
            
            for i, pred in enumerate(predictions):
                result = {
                    "prediction": int(pred),
                    "prediction_label": "fraud" if int(pred) == 1 else "legitimate"
                }
                
                # Add confidence scores
                if probabilities:
                    prob = probabilities[i]
                    result["confidence"] = float(prob[int(pred)])
                    result["probabilities"] = {
                        "legitimate": float(prob[0]),
                        "fraud": float(prob[1])
                    }
                
                # Add metadata if available
                if i < len(metadata_list):
                    result["metadata"] = metadata_list[i]
                
                # Add risk level
                if probabilities:
                    fraud_prob = float(prob[1])
                    if fraud_prob >= 0.8:
                        result["risk_level"] = "high"
                    elif fraud_prob >= 0.5:
                        result["risk_level"] = "medium"
                    else:
                        result["risk_level"] = "low"
                
                processed_outputs.append(result)
            
            # Add transformer metrics
            response = {
                "predictions": processed_outputs,
                "transformer_metrics": {
                    "total_requests": self.request_count,
                    "error_count": self.error_count,
                    "avg_latency_ms": (self.total_latency / self.request_count * 1000) if self.request_count > 0 else 0
                }
            }
            
            return response
            
        except Exception as e:
            raise


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument('--model_dir', 
                   default=os.environ.get('STORAGE_URI'),
                   help='Model directory path')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)
    
    model = AdvancedFraudTransformer(
        name=args.model_name,
        model_dir=args.model_dir,
        predictor_config=PredictorConfig(
            args.predictor_host,
            PredictorProtocol.REST_V2.value,
            args.predictor_use_ssl,
            args.predictor_request_timeout_seconds,
            args.predictor_request_retries,
            args.enable_predictor_health_check,
        ),
    )
    
    model.load()
    
    ModelServer().start([model])
