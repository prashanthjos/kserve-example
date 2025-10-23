"""
Ensemble Aggregator for Multiple Model Predictions
Combines predictions from multiple models using various strategies
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

from kserve import Model, ModelServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleAggregator(Model):
    """
    Aggregates predictions from multiple models.
    
    Supported aggregation methods:
    - majority_vote: Simple majority voting
    - weighted_average: Weighted average of probabilities
    - max_confidence: Select prediction with highest confidence
    - stacking: Use a meta-model (future enhancement)
    """
    
    def __init__(
        self,
        name: str,
        aggregation_method: str = "weighted_average",
        model_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(name)
        self.aggregation_method = aggregation_method
        self.model_weights = model_weights or {}
        self.ready = True

    def load(self):
        """Initialize aggregator."""
        logger.info(f"Ensemble aggregator initialized with method: {self.aggregation_method}")
        self.ready = True
        return self.ready

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Aggregate predictions from multiple models.
        
        Expected input format:
        {
            "predictions": {
                "model_1": {"predictions": [...], "probabilities": [...]},
                "model_2": {"predictions": [...], "probabilities": [...]},
                ...
            }
        }
        
        Returns:
            Aggregated predictions
        """
        try:
            model_predictions = payload.get("predictions", {})
            
            if not model_predictions:
                raise ValueError("No model predictions provided")
            
            # Aggregate based on selected method
            if self.aggregation_method == "majority_vote":
                result = self._majority_vote(model_predictions)
            elif self.aggregation_method == "weighted_average":
                result = self._weighted_average(model_predictions)
            elif self.aggregation_method == "max_confidence":
                result = self._max_confidence(model_predictions)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
            # Add metadata
            result["aggregation_metadata"] = {
                "method": self.aggregation_method,
                "num_models": len(model_predictions),
                "models": list(model_predictions.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Aggregation error: {str(e)}")
            raise

    def _majority_vote(self, model_predictions: Dict) -> Dict:
        """
        Aggregate using majority voting.
        
        Args:
            model_predictions: Dictionary of model predictions
            
        Returns:
            Aggregated predictions
        """
        num_instances = len(next(iter(model_predictions.values()))["predictions"])
        aggregated_predictions = []
        
        for i in range(num_instances):
            votes = []
            for model_name, preds in model_predictions.items():
                pred = preds["predictions"][i]
                if isinstance(pred, dict):
                    pred = pred.get("prediction", pred.get("class", 0))
                votes.append(int(pred))
            
            # Get majority vote
            majority = max(set(votes), key=votes.count)
            confidence = votes.count(majority) / len(votes)
            
            aggregated_predictions.append({
                "prediction": majority,
                "confidence": confidence,
                "votes": {f"model_{j}": v for j, v in enumerate(votes)}
            })
        
        return {"predictions": aggregated_predictions}

    def _weighted_average(self, model_predictions: Dict) -> Dict:
        """
        Aggregate using weighted average of probabilities.
        
        Args:
            model_predictions: Dictionary of model predictions
            
        Returns:
            Aggregated predictions
        """
        num_instances = len(next(iter(model_predictions.values()))["predictions"])
        aggregated_predictions = []
        
        # Get weights for each model
        weights = {}
        total_weight = 0.0
        for model_name in model_predictions.keys():
            weight = self.model_weights.get(model_name, 1.0)
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        for i in range(num_instances):
            weighted_probs = None
            model_contributions = {}
            
            for model_name, preds in model_predictions.items():
                weight = weights[model_name]
                
                # Extract probabilities
                if "probabilities" in preds:
                    probs = np.array(preds["probabilities"][i])
                else:
                    # If no probabilities, use one-hot encoding of prediction
                    pred = preds["predictions"][i]
                    if isinstance(pred, dict):
                        pred = pred.get("prediction", 0)
                    probs = np.array([1.0 if j == pred else 0.0 for j in range(2)])
                
                # Weighted sum
                if weighted_probs is None:
                    weighted_probs = weight * probs
                else:
                    weighted_probs += weight * probs
                
                model_contributions[model_name] = {
                    "weight": float(weight),
                    "probabilities": probs.tolist()
                }
            
            # Final prediction
            final_prediction = int(np.argmax(weighted_probs))
            confidence = float(weighted_probs[final_prediction])
            
            aggregated_predictions.append({
                "prediction": final_prediction,
                "confidence": confidence,
                "probabilities": weighted_probs.tolist(),
                "model_contributions": model_contributions
            })
        
        return {"predictions": aggregated_predictions}

    def _max_confidence(self, model_predictions: Dict) -> Dict:
        """
        Select prediction with maximum confidence across models.
        
        Args:
            model_predictions: Dictionary of model predictions
            
        Returns:
            Aggregated predictions
        """
        num_instances = len(next(iter(model_predictions.values()))["predictions"])
        aggregated_predictions = []
        
        for i in range(num_instances):
            max_confidence = -1.0
            best_prediction = None
            best_model = None
            all_model_preds = {}
            
            for model_name, preds in model_predictions.items():
                pred_data = preds["predictions"][i]
                
                if isinstance(pred_data, dict):
                    prediction = pred_data.get("prediction", 0)
                    confidence = pred_data.get("confidence", 0.5)
                else:
                    prediction = int(pred_data)
                    # Get confidence from probabilities if available
                    if "probabilities" in preds:
                        probs = preds["probabilities"][i]
                        confidence = float(max(probs))
                    else:
                        confidence = 0.5
                
                all_model_preds[model_name] = {
                    "prediction": prediction,
                    "confidence": confidence
                }
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_prediction = prediction
                    best_model = model_name
            
            aggregated_predictions.append({
                "prediction": best_prediction,
                "confidence": max_confidence,
                "selected_model": best_model,
                "all_predictions": all_model_preds
            })
        
        return {"predictions": aggregated_predictions}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='fraud-detection-aggregator',
                       help='Name of the aggregator')
    parser.add_argument('--aggregation_method', default='weighted_average',
                       choices=['majority_vote', 'weighted_average', 'max_confidence'],
                       help='Aggregation method')
    parser.add_argument('--http_port', type=int, default=8080,
                       help='HTTP port')
    
    args = parser.parse_args()
    
    # Example weights (can be loaded from config)
    model_weights = {
        "random-forest": 0.4,
        "xgboost": 0.4,
        "neural-network": 0.2
    }
    
    model = EnsembleAggregator(
        name=args.model_name,
        aggregation_method=args.aggregation_method,
        model_weights=model_weights
    )
    
    ModelServer(http_port=args.http_port).start([model])
