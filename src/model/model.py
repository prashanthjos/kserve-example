import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class FraudDetectionModel:
    """Fraud detection model implementation."""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """Initialize the model with hyperparameters."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X_train, y_train, feature_names=None):
        """Train the model on the provided data."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        return self
    
    def predict(self, X):
        """Make predictions on input data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        return metrics
    
    def save(self, model_path):
        """Save the model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create model config
        model_config = {
            'feature_names': self.feature_names,
            'model_type': 'RandomForestClassifier',
            'params': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'random_state': self.model.random_state
            }
        }
        
        # Save model and config
        joblib.dump(self.model, f"{model_path}/model.joblib")
        
        with open(f"{model_path}/model_config.json", 'w') as f:
            json.dump(model_config, f)
        
        return model_path
    
    @classmethod
    def load(cls, model_path):
        """Load a trained model from disk."""
        # Load the sklearn model
        model_file = f"{model_path}/model.joblib"
        sklearn_model = joblib.load(model_file)
        
        # Load the config
        with open(f"{model_path}/model_config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            n_estimators=config['params']['n_estimators'],
            max_depth=config['params']['max_depth'],
            random_state=config['params']['random_state']
        )
        
        # Set the model and mark as trained
        instance.model = sklearn_model
        instance.feature_names = config['feature_names']
        instance.is_trained = True
        
        return instance