"""
Data loading and preprocessing module.
Handles CSV uploads, missing values, and train/test splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional


class DataProcessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = None
        self.is_fitted = False
        
    def load_and_validate(self, uploaded_file) -> pd.DataFrame:
        """
        Load CSV file and perform basic validation.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            ValueError: If file is empty or invalid
        """
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                raise ValueError("Uploaded file is empty")
                
            if len(df) < 10:
                raise ValueError("Dataset too small. Need at least 10 rows.")
                
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def get_column_info(self, df: pd.DataFrame) -> dict:
        """
        Analyze dataframe and suggest target/feature columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Column information including suggestions
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        suggested_target = None
        if categorical_cols:
            cat_unique_counts = {col: df[col].nunique() for col in categorical_cols}
            suggested_target = min(cat_unique_counts, key=cat_unique_counts.get)
        else:
            suggested_target = df.columns[-1]
        
        return {
            'all_columns': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'suggested_target': suggested_target,
            'missing_values': df.isnull().sum().to_dict(),
            'shape': df.shape
        }
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            df: Input dataframe
            strategy: 'drop', 'mean', or 'median'
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df.dropna()
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            return df.dropna()
        else:
            return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        Handles categorical encoding and feature selection.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            feature_columns: List of feature column names (optional)
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        self.target_column = target_column
        
        y = df[target_column].copy()
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)
            self.is_fitted = True
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.feature_columns = feature_columns
        X = df[feature_columns].copy()
        
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
            self.feature_columns = X.columns.tolist()
        
        return X, pd.Series(y, index=X.index)
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and validation sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
    
    def get_class_names(self) -> List[str]:
        """Get original class names if target was encoded."""
        if self.is_fitted:
            return self.label_encoder.classes_.tolist()
        return []
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original values."""
        if self.is_fitted:
            return self.label_encoder.inverse_transform(y_encoded)
        return y_encoded