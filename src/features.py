"""
Feature engineering and preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, List

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


class FeaturePreprocessor:
    """Handles feature encoding and scaling."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.numeric_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.fitted = False
    
    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: DataFrame with features
            
        Returns:
            self for chaining
        """
        # Fit numeric scaler
        self.numeric_scaler.fit(X[NUMERIC_FEATURES])
        
        # Fit categorical encoders
        for cat_feat in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(X[cat_feat].values)
            self.categorical_encoders[cat_feat] = le
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Transformed numpy array
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Transform numeric features
        X_numeric = self.numeric_scaler.transform(X[NUMERIC_FEATURES])
        
        # Transform categorical features
        X_categorical_list = []
        for cat_feat in CATEGORICAL_FEATURES:
            encoder = self.categorical_encoders[cat_feat]
            encoded = encoder.transform(X[cat_feat].values).reshape(-1, 1)
            X_categorical_list.append(encoded)
        
        # Combine all transformations
        if X_categorical_list:
            X_categorical = np.hstack(X_categorical_list)
            X_transformed = np.hstack([X_numeric, X_categorical])
        else:
            X_transformed = X_numeric
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional derived features for richer representations.
    
    Args:
        df: Original feature DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()
    
    # Interaction features
    df_eng["age_x_coverage"] = df_eng["printer_age"] * df_eng["coverage_pct"]
    df_eng["humidity_temp_distance"] = np.abs(df_eng["humidity"] - 50) + np.abs(df_eng["temperature"] - 22)
    df_eng["viscosity_distance_from_optimal"] = np.abs(df_eng["ink_viscosity"] - 35)
    
    # Categorical encodings for better interpretability
    df_eng["head_type_piezo"] = (df_eng["head_type"] == "piezo").astype(int)
    
    # Environmental stress index (simplified)
    df_eng["env_stress_index"] = (
        (np.abs(df_eng["humidity"] - 50) / 20) +
        (np.abs(df_eng["temperature"] - 22) / 5)
    ) / 2
    
    return df_eng


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for model training.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    X = df[feature_cols]
    y = df[target_col]
    
    preprocessor = FeaturePreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed, y.values, preprocessor
