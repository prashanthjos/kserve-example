"""
Model Monitoring Component for KServe
Features:
- Request/response logging
- Performance metrics collection
- Data drift detection
- Prediction distribution tracking
- Alerting thresholds
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque, defaultdict
import time

from kserve import Model, ModelServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor(Model):
    """
    Monitoring component for tracking model performance and data quality.
    
    Features:
    - Request/response logging
    - Latency tracking
    - Prediction distribution monitoring
    - Data drift detection
    - Performance metrics
    """
    
    def __init__(
        self,
        name: str,
        predictor_host: str,
        window_size: int = 1000,
        drift_threshold: float = 0.1
    ):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Monitoring data structures
        self.request_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        
        # Metrics
        self.total_requests = 0
        self.total_predictions = {0: 0, 1: 0}
        self.error_count = 0
        
        # Feature statistics for drift detection
        self.baseline_stats = None
        self.current_stats = defaultdict(list)
        
        # Alert thresholds
        self.high_latency_threshold = 1.0  # seconds
        self.high_fraud_rate_threshold = 0.3  # 30% fraud rate
        
        self.ready = True

    def load(self):
        """Initialize monitoring component."""
        logger.info("Model monitor initialized")
        self.ready = True
        return self.ready

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Monitor and log incoming requests.
        
        Args:
            inputs: Input data
            headers: Request headers
            
        Returns:
            Unmodified inputs (pass-through)
        """
        try:
            self.total_requests += 1
            
            # Log request metadata
            request_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': self.total_requests,
                'num_instances': len(inputs.get('instances', []))
            }
            
            self.request_history.append(request_data)
            
            # Track feature statistics for drift detection
            instances = inputs.get('instances', [])
            if instances:
                self._update_feature_stats(instances)
            
            # Add monitoring metadata to the request
            if 'metadata' not in inputs:
                inputs['metadata'] = {}
            
            inputs['metadata']['monitor_request_id'] = self.total_requests
            inputs['metadata']['monitor_timestamp'] = request_data['timestamp']
            
            return inputs
            
        except Exception as e:
            logger.error(f"Monitoring preprocessing error: {str(e)}")
            # Don't fail the request due to monitoring errors
            return inputs

    def postprocess(self, outputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Monitor and analyze predictions.
        
        Args:
            outputs: Model predictions
            headers: Response headers
            
        Returns:
            Outputs with monitoring metadata
        """
        try:
            # Extract predictions
            predictions = outputs.get('predictions', [])
            
            # Track predictions
            for pred in predictions:
                if isinstance(pred, dict):
                    pred_class = pred.get('prediction', 0)
                else:
                    pred_class = int(pred)
                
                self.total_predictions[pred_class] = self.total_predictions.get(pred_class, 0) + 1
                self.prediction_history.append({
                    'prediction': pred_class,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Check for alerts
            alerts = self._check_alerts(metrics)
            
            # Add monitoring data to response
            outputs['monitoring'] = {
                'metrics': metrics,
                'alerts': alerts,
                'drift_detected': self._detect_drift()
            }
            
            logger.info(f"Monitored {len(predictions)} predictions. Fraud rate: {metrics['fraud_rate']:.2%}")
            
            if alerts:
                logger.warning(f"Alerts triggered: {alerts}")
            
            return outputs
            
        except Exception as e:
            logger.error(f"Monitoring postprocessing error: {str(e)}")
            # Don't fail the request due to monitoring errors
            return outputs

    def _update_feature_stats(self, instances: List):
        """
        Update feature statistics for drift detection.
        
        Args:
            instances: List of input instances
        """
        try:
            # Convert instances to numpy array
            data = np.array(instances)
            
            # Update current window statistics
            for col_idx in range(data.shape[1]):
                feature_values = data[:, col_idx]
                self.current_stats[f'feature_{col_idx}'].extend(feature_values.tolist())
                
                # Keep only recent values
                if len(self.current_stats[f'feature_{col_idx}']) > self.window_size:
                    self.current_stats[f'feature_{col_idx}'] = \
                        self.current_stats[f'feature_{col_idx}'][-self.window_size:]
            
            # Set baseline if not already set
            if self.baseline_stats is None and self.total_requests > 100:
                self._set_baseline()
                
        except Exception as e:
            logger.error(f"Error updating feature stats: {str(e)}")

    def _set_baseline(self):
        """Set baseline statistics for drift detection."""
        try:
            self.baseline_stats = {}
            
            for feature_name, values in self.current_stats.items():
                if len(values) > 0:
                    self.baseline_stats[feature_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            
            logger.info("Baseline statistics set for drift detection")
            
        except Exception as e:
            logger.error(f"Error setting baseline: {str(e)}")

    def _detect_drift(self) -> Dict:
        """
        Detect data drift using statistical tests.
        
        Returns:
            Dictionary with drift detection results
        """
        if self.baseline_stats is None:
            return {'detected': False, 'reason': 'baseline not yet established'}
        
        try:
            drift_features = []
            
            for feature_name, baseline in self.baseline_stats.items():
                current_values = self.current_stats.get(feature_name, [])
                
                if len(current_values) < 50:
                    continue
                
                # Calculate current statistics
                current_mean = np.mean(current_values)
                current_std = np.std(current_values)
                
                # Check for significant drift (simple threshold-based)
                mean_diff = abs(current_mean - baseline['mean'])
                std_diff = abs(current_std - baseline['std'])
                
                # Normalize by baseline std to get relative change
                if baseline['std'] > 0:
                    normalized_mean_diff = mean_diff / baseline['std']
                    if normalized_mean_diff > self.drift_threshold:
                        drift_features.append({
                            'feature': feature_name,
                            'baseline_mean': baseline['mean'],
                            'current_mean': current_mean,
                            'drift_magnitude': normalized_mean_diff
                        })
            
            if drift_features:
                return {
                    'detected': True,
                    'drifted_features': drift_features,
                    'num_drifted': len(drift_features)
                }
            else:
                return {'detected': False}
                
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            return {'detected': False, 'error': str(e)}

    def _calculate_metrics(self) -> Dict:
        """
        Calculate monitoring metrics.
        
        Returns:
            Dictionary of metrics
        """
        total_preds = sum(self.total_predictions.values())
        
        metrics = {
            'total_requests': self.total_requests,
            'total_predictions': total_preds,
            'error_count': self.error_count,
            'fraud_count': self.total_predictions.get(1, 0),
            'legitimate_count': self.total_predictions.get(0, 0),
            'fraud_rate': self.total_predictions.get(1, 0) / total_preds if total_preds > 0 else 0.0
        }
        
        # Calculate latency metrics if available
        if self.latency_history:
            metrics['avg_latency_ms'] = np.mean(list(self.latency_history)) * 1000
            metrics['p95_latency_ms'] = np.percentile(list(self.latency_history), 95) * 1000
            metrics['p99_latency_ms'] = np.percentile(list(self.latency_history), 99) * 1000
        
        # Calculate recent fraud rate (last 100 predictions)
        recent_predictions = list(self.prediction_history)[-100:]
        if recent_predictions:
            recent_fraud_count = sum(1 for p in recent_predictions if p['prediction'] == 1)
            metrics['recent_fraud_rate'] = recent_fraud_count / len(recent_predictions)
        else:
            metrics['recent_fraud_rate'] = 0.0
        
        return metrics

    def _check_alerts(self, metrics: Dict) -> List[Dict]:
        """
        Check if any alert thresholds are exceeded.
        
        Args:
            metrics: Current metrics
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Check fraud rate
        if metrics.get('recent_fraud_rate', 0) > self.high_fraud_rate_threshold:
            alerts.append({
                'type': 'HIGH_FRAUD_RATE',
                'severity': 'warning',
                'message': f"Recent fraud rate ({metrics['recent_fraud_rate']:.1%}) exceeds threshold ({self.high_fraud_rate_threshold:.1%})",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Check latency
        if metrics.get('p95_latency_ms', 0) > self.high_latency_threshold * 1000:
            alerts.append({
                'type': 'HIGH_LATENCY',
                'severity': 'warning',
                'message': f"P95 latency ({metrics['p95_latency_ms']:.0f}ms) exceeds threshold ({self.high_latency_threshold*1000}ms)",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Check error rate
        error_rate = metrics['error_count'] / metrics['total_requests'] if metrics['total_requests'] > 0 else 0
        if error_rate > 0.05:  # 5% error rate
            alerts.append({
                'type': 'HIGH_ERROR_RATE',
                'severity': 'critical',
                'message': f"Error rate ({error_rate:.1%}) exceeds 5%",
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return alerts

    def get_metrics(self) -> Dict:
        """
        Get current monitoring metrics.
        
        Returns:
            Current metrics and statistics
        """
        metrics = self._calculate_metrics()
        drift_info = self._detect_drift()
        
        return {
            'metrics': metrics,
            'drift': drift_info,
            'window_size': self.window_size,
            'baseline_set': self.baseline_stats is not None
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='fraud-detection-monitor',
                       help='Name of the monitor')
    parser.add_argument('--predictor_host', required=True,
                       help='Predictor host address')
    parser.add_argument('--http_port', type=int, default=8080,
                       help='HTTP port')
    parser.add_argument('--window_size', type=int, default=1000,
                       help='Monitoring window size')
    
    args = parser.parse_args()
    
    model = ModelMonitor(
        name=args.model_name,
        predictor_host=args.predictor_host,
        window_size=args.window_size
    )
    
    ModelServer(http_port=args.http_port).start([model])
