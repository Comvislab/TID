# -*- coding: utf-8 -*-
"""
Feature Preprocessing

Normalization and scaling of features.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def normalize_features(X_train: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize training features using MinMaxScaler.
    
    Uses MinMaxScaler as in the original implementation.
    The scaler is fit on training data and returns both transformed data
    and the scaler object for later use on test data.
    
    Args:
        X_train: Training feature matrix
    
    Returns:
        Tuple of (normalized X_train, fitted scaler)
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_normalized = scaler.transform(X_train)
    
    return X_normalized, scaler
