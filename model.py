import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MLModel:
    """Wrapper for ML classification models with training and evaluation."""
    
    def __init__(self, model_type: str = 'logistic_regression', hyperparameters: Optional[Dict] = None):
        """
        Initialize model.
        
        Args:
            model_type: One of 'logistic_regression', 'random_forest', 'xgboost'
            hyperparameters: Dict of hyperparameters for the model
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.n_classes = 0
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sklearn/xgboost model based on type."""
        if self.model_type == 'logistic_regression':
            default_params = {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs'
            }
            params = {**default_params, **self.hyperparameters}
            self.model = LogisticRegression(**params)
            
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **self.hyperparameters}
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            params = {**default_params, **self.hyperparameters}
            self.model = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dict containing training metadata
        """
        self.feature_names = X_train.columns.tolist()
        self.n_classes = len(np.unique(y_train))
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        return {
            'success': True,
            'n_samples': len(X_train),
            'n_features': len(self.feature_names),
            'n_classes': self.n_classes,
            'train_metrics': train_metrics
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on validation/test data.
        
        Args:
            X: Feature matrix
            y_true: True labels
            
        Returns:
            Dict containing all evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        misclassified_idx = np.where(y_true != y_pred)[0]
        metrics['misclassified_indices'] = misclassified_idx
        metrics['n_misclassified'] = len(misclassified_idx)
        
        return metrics
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        if y_proba is not None:
            try:
                if self.n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba, 
                        multi_class='ovr', 
                        average='weighted'
                    )
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):

            if self.n_classes == 2:
                importances = np.abs(self.model.coef_[0])
            else:
                importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            raise ValueError(f"Model type {self.model_type} does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_prediction_confidence(
        self, 
        X: pd.DataFrame, 
        y_true: pd.Series
    ) -> pd.DataFrame:
        """
        Get prediction confidence for each sample.
        
        Args:
            X: Feature matrix
            y_true: True labels
            
        Returns:
            DataFrame with confidence scores and correctness
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        confidence = np.max(y_proba, axis=1)
        
        confidence_df = pd.DataFrame({
            'true_label': y_true.values,
            'predicted_label': y_pred,
            'confidence': confidence,
            'correct': (y_true.values == y_pred)
        })
        
        return confidence_df
    
    def get_misclassified_samples(
        self,
        X: pd.DataFrame,
        y_true: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Extract misclassified samples with predictions.
        
        Returns:
            Tuple of (X_misclassified, y_true_misclassified, y_pred_misclassified)
        """
        y_pred = self.predict(X)
        misclassified_mask = y_true.values != y_pred
        
        X_misclassified = X[misclassified_mask]
        y_true_misclassified = y_true[misclassified_mask]
        y_pred_misclassified = y_pred[misclassified_mask]
        
        return X_misclassified, y_true_misclassified, y_pred_misclassified


def calculate_feature_importance_drift(
    model: MLModel,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series
) -> pd.DataFrame:
    """
    Calculate feature importance on both train and validation sets
    to detect importance drift.
    
    Args:
        model: Trained MLModel instance
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        
    Returns:
        DataFrame comparing train vs validation importance
    """
    train_importance = model.get_feature_importance()
    
    val_model = MLModel(
        model_type=model.model_type,
        hyperparameters=model.hyperparameters
    )
    val_model.train(X_val, y_val)
    val_importance = val_model.get_feature_importance()
    
    comparison = train_importance.merge(
        val_importance,
        on='feature',
        suffixes=('_train', '_val')
    )
    
    comparison['drift'] = np.abs(
        comparison['importance_train'] - comparison['importance_val']
    )
    
    comparison['drift_normalized'] = comparison['drift'] / (
        (comparison['importance_train'] + comparison['importance_val']) / 2 + 1e-10
    )
    
    return comparison.sort_values('drift', ascending=False)