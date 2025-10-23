"""
Advanced Model Explainer for Fraud Detection
Features:
- SHAP TreeExplainer integration
- Multiple explanation formats
- Visualization data generation
- Feature importance ranking
- Explanation caching for performance
"""

import os
import json
import joblib
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import time

try:
    import shap
except ImportError:
    raise ImportError("SHAP library required. Install with: pip install shap")

from kserve import Model, ModelServer
from kserve.storage import Storage
from kserve.errors import InvalidInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFraudExplainer(Model):
    """
    Advanced explainer with comprehensive SHAP integration.
    
    Features:
    - SHAP value computation
    - Feature importance ranking
    - Explanation visualization data
    - Performance optimization with caching
    - Multiple explanation formats
    """
    
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        
        # Model artifacts
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.scaler = None
        
        # Explanation settings
        self.max_display_features = 10
        self.explanation_format = "detailed"  # detailed, summary, or minimal
        
        # Performance metrics
        self.explanation_count = 0
        self.total_explanation_time = 0.0
        
        
        self.ready = False

    def load(self):
        """Load model and initialize SHAP explainer."""
        try:
            model_dir = Storage.download(self.model_dir)
            
            if not os.path.exists(model_dir):
                self.ready = False
                return False
            
            # Try multiple paths for the model file
            # The pipeline saves models in different locations depending on configuration
            possible_model_paths = [
                os.path.join(model_dir, "pred-model", "model.joblib"),  # Current pipeline structure
                os.path.join(model_dir, "model", "model.joblib"),       # Alternative structure
                os.path.join(model_dir, "model.joblib"),                # Root directory fallback
            ]
            
            model_path = None
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                # Check subdirectories
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path):
                        subdir_contents = os.listdir(item_path)
                self.ready = False
                return False
            
            self.model = joblib.load(model_path)
            
            # Load feature names from root directory
            feature_names_path = os.path.join(model_dir, "feature_names.json")
            if not os.path.exists(feature_names_path):
                self.ready = False
                return False
            
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            
            # Load scaler if available (from root directory)
            try:
                scaler_path = os.path.join(model_dir, "scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                else:
                    self.scaler = None
            except Exception as e:
                self.scaler = None
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            self.ready = True
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error loading model: {str(e)}"
            traceback_str = traceback.format_exc()
            self.ready = False
            # Don't raise - let KServe handle it gracefully
            return False

    def explain(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Generate comprehensive explanations for predictions.
        
        Args:
            payload: Input data with 'instances' key
            headers: Optional request headers
            
        Returns:
            Detailed explanations with SHAP values and feature importance
        """
        start_time = time.time()
        self.explanation_count += 1
        
        try:
            instances = payload.get("instances", [])
            if not instances:
                raise InvalidInput("No instances provided in request")
            
            # Convert to numpy array
            instances_array = np.asarray(instances)
            
            # Validate input shape
            expected_features = len(self.feature_names)
            if instances_array.shape[1] != expected_features:
                raise InvalidInput(
                    f"Expected {expected_features} features, got {instances_array.shape[1]}"
                )
            
            # Generate SHAP values
            logger.info(f"Computing SHAP values for {len(instances)} instances...")
            shap_values = self.explainer.shap_values(instances_array)
            
            # Handle different SHAP value formats
            # For binary classification, shap_values might be a list [class0, class1]
            if isinstance(shap_values, list):
                # Focus on positive class (fraud) for binary classification
                shap_values_fraud = shap_values[1]
                expected_values = self.explainer.expected_value[1] if isinstance(
                    self.explainer.expected_value, list
                ) else self.explainer.expected_value
            else:
                shap_values_fraud = shap_values
                # Extract expected value for fraud class (index 1) if it's an array
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    expected_values = self.explainer.expected_value[1]
                else:
                    expected_values = self.explainer.expected_value
            
            # Get model predictions for context
            predictions = self.model.predict(instances_array)
            probabilities = self.model.predict_proba(instances_array)
            
            # Generate explanations for each instance
            explanations = []
            for i, instance in enumerate(instances_array):
                explanation = self._generate_instance_explanation(
                    instance=instance,
                    shap_values=shap_values_fraud[i],
                    expected_value=expected_values,
                    prediction=predictions[i],
                    probabilities=probabilities[i]
                )
                explanations.append(explanation)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            self.total_explanation_time += elapsed_time
            
            logger.info(f"Generated {len(explanations)} explanations in {elapsed_time:.4f}s")
            
            # Return comprehensive response
            return {
                "explanations": explanations,
                "metadata": {
                    "model_type": type(self.model).__name__,
                    "explainer_type": "SHAP TreeExplainer",
                    "num_features": len(self.feature_names),
                    "timestamp": datetime.utcnow().isoformat(),
                    "explanation_time_ms": elapsed_time * 1000,
                    "total_explanations": self.explanation_count
                }
            }
            
        except InvalidInput:
            raise
        except Exception as e:
            logger.error(f"Explanation error: {str(e)}")
            raise InvalidInput(f"Failed to generate explanation: {str(e)}")

    def _generate_instance_explanation(
        self,
        instance: np.ndarray,
        shap_values: np.ndarray,
        expected_value: float,
        prediction: int,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Generate detailed explanation for a single instance.
        
        Args:
            instance: Input feature values
            shap_values: SHAP values for this instance
            expected_value: Model's expected/base value
            prediction: Model prediction
            probabilities: Prediction probabilities
            
        Returns:
            Comprehensive explanation dictionary
        """
        # Create feature importance ranking
        feature_importance = []
        for i, (feature_name, shap_val, feature_val) in enumerate(
            zip(self.feature_names, shap_values, instance)
        ):
            # Handle case where shap_val might be an array (for binary classification)
            if isinstance(shap_val, np.ndarray) and shap_val.shape:
                shap_val_scalar = float(shap_val[1] if len(shap_val) > 1 else shap_val[0])
            else:
                shap_val_scalar = float(shap_val)
            
            feature_importance.append({
                "feature": feature_name,
                "value": float(feature_val),
                "shap_value": shap_val_scalar,
                "abs_shap_value": abs(shap_val_scalar),
                "contribution": "increases fraud risk" if shap_val_scalar > 0 else "decreases fraud risk"
            })
        
        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        # Get top contributing features
        top_features = feature_importance[:self.max_display_features]
        
        # Calculate prediction details
        fraud_probability = float(probabilities[1])
        legitimate_probability = float(probabilities[0])
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            prediction=prediction,
            fraud_prob=fraud_probability,
            top_features=top_features[:3]
        )
        
        # Create visualization data
        visualization_data = self._create_visualization_data(
            feature_importance=feature_importance,
            expected_value=expected_value
        )
        
        return {
            "prediction": {
                "class": int(prediction),
                "label": "fraud" if prediction == 1 else "legitimate",
                "probabilities": {
                    "legitimate": legitimate_probability,
                    "fraud": fraud_probability
                },
                "confidence": float(max(probabilities))
            },
            "explanation": {
                "base_value": float(expected_value),
                "prediction_value": float(expected_value + sum([f['shap_value'] for f in feature_importance])),
                "text": explanation_text,
                "top_features": top_features,
                "all_features": feature_importance
            },
            "shap_details": {
                "expected_value": float(expected_value),
                "shap_values": [f['shap_value'] for f in feature_importance],
                "feature_values": instance.tolist(),
                "feature_names": self.feature_names
            },
            "visualization": visualization_data
        }

    def _generate_explanation_text(
        self,
        prediction: int,
        fraud_prob: float,
        top_features: List[Dict]
    ) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            prediction: Model prediction
            fraud_prob: Fraud probability
            top_features: Top contributing features
            
        Returns:
            Natural language explanation
        """
        if prediction == 1:
            text = f"This transaction is predicted as FRAUD with {fraud_prob:.1%} confidence. "
        else:
            text = f"This transaction is predicted as LEGITIMATE with {(1-fraud_prob):.1%} confidence. "
        
        text += "The most influential factors are:\n"
        
        for i, feat in enumerate(top_features, 1):
            impact = "increases" if feat['shap_value'] > 0 else "decreases"
            text += f"{i}. {feat['feature']} (value: {feat['value']:.3f}) {impact} fraud risk by {abs(feat['shap_value']):.3f}\n"
        
        return text

    def _create_visualization_data(
        self,
        feature_importance: List[Dict],
        expected_value: float
    ) -> Dict:
        """
        Create data for visualization (e.g., for force plots, waterfall charts).
        
        Args:
            feature_importance: Sorted feature importance list
            expected_value: Base/expected value
            
        Returns:
            Data formatted for visualization libraries
        """
        # Prepare data for waterfall chart
        waterfall_data = {
            "features": [f['feature'] for f in feature_importance[:self.max_display_features]],
            "shap_values": [f['shap_value'] for f in feature_importance[:self.max_display_features]],
            "feature_values": [f['value'] for f in feature_importance[:self.max_display_features]],
            "base_value": float(expected_value)
        }
        
        # Prepare data for force plot
        force_plot_data = {
            "base_value": float(expected_value),
            "shap_values": [f['shap_value'] for f in feature_importance],
            "features": [f['feature'] for f in feature_importance],
            "feature_values": [f['value'] for f in feature_importance]
        }
        
        # Bar chart data
        bar_chart_data = {
            "features": [f['feature'] for f in feature_importance[:self.max_display_features]],
            "importance": [f['abs_shap_value'] for f in feature_importance[:self.max_display_features]]
        }
        
        return {
            "waterfall": waterfall_data,
            "force_plot": force_plot_data,
            "bar_chart": bar_chart_data
        }

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Alias for explain method to maintain compatibility.
        """
        return self.explain(payload, headers)


if __name__ == "__main__":
    import argparse
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='fraud-detection-explainer',
                       help='Name of the explainer')
    parser.add_argument('--predictor_host',
                       help='Predictor host (injected by KServe, not used)')
    parser.add_argument('--model_dir', 
                       default=os.environ.get('STORAGE_URI'),
                       help='Model directory path (can use STORAGE_URI env var)')
    parser.add_argument('--http_port', type=int, default=8080,
                       help='HTTP port')
    
    args = parser.parse_args()
    
    
    if not args.model_dir:
        raise ValueError("model_dir must be provided via --model_dir or STORAGE_URI env var")
    
    model = AdvancedFraudExplainer(
        name=args.model_name,
        model_dir=args.model_dir
    )
    
    model.load()
    
    ModelServer(http_port=args.http_port).start([model])
