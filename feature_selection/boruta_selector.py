# -*- coding: utf-8 -*-
"""
Boruta Feature Selection

Uses Boruta algorithm with Random Forest for feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from typing import Callable

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BORUTA_RF_CONFIG, SEED


def boruta_select(
    X_normalized: np.ndarray,
    y: np.ndarray,
    columns: list,
    k: int,
    figures_dir: str = None,
    plot_func: Callable = None
) -> np.ndarray:
    """
    Select top-k features using Boruta algorithm with Random Forest.
    
    Boruta wraps a Random Forest classifier and uses shadow features
    to determine feature importance.
    
    Args:
        X_normalized: Normalized feature matrix
        y: Target labels
        columns: Original feature column names
        k: Number of features to select
        figures_dir: Directory to save rank plots
        plot_func: Optional plotting function
    
    Returns:
        Array of selected feature indices
    """
    # Create Random Forest classifier with configured parameters
    rf_model = RandomForestClassifier(**BORUTA_RF_CONFIG)
    
    # Create and fit Boruta selector
    boruta_selector = BorutaPy(
        rf_model, 
        n_estimators='auto', 
        verbose=5, 
        random_state=SEED
    )
    boruta_selector.fit(X_normalized, y)
    
    # Create ranking DataFrame
    boruta_scores = pd.DataFrame(boruta_selector.ranking_)
    boruta_columns = pd.DataFrame(columns)
    
    feature_scores = pd.concat([boruta_columns, boruta_scores], axis=1)
    feature_scores.columns = ['Boruta_Specs', 'Boruta_Scores']
    
    # Get top-k features (lowest ranking = most important)
    best_features = feature_scores.nsmallest(k, 'Boruta_Scores')
    selected_indices = best_features.index.values
    
    # Plot if function provided
    if plot_func is not None and figures_dir is not None:
        plot_func("BorutaKBest (n_largest are selected)", boruta_selector.ranking_, figures_dir)
    
    return selected_indices
