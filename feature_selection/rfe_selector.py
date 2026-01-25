# -*- coding: utf-8 -*-
"""
Recursive Feature Elimination (RFE)

RFE with multiple classifier types: RFC, ETC, SVC, DTC.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Callable, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RFE_CLASSIFIERS


def get_rfe_models() -> Dict[str, Any]:
    """
    Create classifier models for RFE with configured parameters.
    
    Returns:
        Dictionary mapping model abbreviations to model instances
    """
    models = {
        'RFC': RandomForestClassifier(**RFE_CLASSIFIERS['RFC']),
        'ETC': ExtraTreesClassifier(**RFE_CLASSIFIERS['ETC']),
        'SVC': SVC(**RFE_CLASSIFIERS['SVC']),
        'DTC': DecisionTreeClassifier(**RFE_CLASSIFIERS['DTC']),
    }
    return models


def rfe_select(
    X_normalized: np.ndarray,
    y: np.ndarray,
    columns: list,
    model: Any,
    model_name: str,
    k: int,
    figures_dir: str = None,
    plot_func: Callable = None
) -> np.ndarray:
    """
    Select top-k features using RFE with specified model.
    
    Args:
        X_normalized: Normalized feature matrix
        y: Target labels
        columns: Original feature column names
        model: Classifier model for RFE
        model_name: Name/abbreviation of the model
        k: Number of features to select
        figures_dir: Directory to save rank plots
        plot_func: Optional plotting function
    
    Returns:
        Array of selected feature indices
    """
    # Initialize and fit RFE
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe.fit(X_normalized, y)
    
    # Get feature rankings
    feature_ranking = rfe.ranking_
    
    # Create ranking DataFrame
    df_ranking = pd.DataFrame(feature_ranking)
    df_columns = pd.DataFrame(columns)
    
    feature_rankings = pd.concat([df_columns, df_ranking], axis=1)
    feature_rankings.columns = [f'{model_name}BestSpecs', f'{model_name}BestRankings']
    
    # Get top-k features (lowest ranking = most important)
    selected_features = feature_rankings.nsmallest(k, f'{model_name}BestRankings').index.values
    
    print(f"#####>{model_name}<#####")
    print(feature_rankings.nsmallest(k, f'{model_name}BestRankings'))
    
    # Plot if function provided
    if plot_func is not None and figures_dir is not None:
        plot_func(f"RFE {model_name}", feature_ranking, figures_dir)
    
    return selected_features


def rfe_select_all(
    X_normalized: np.ndarray,
    y: np.ndarray,
    columns: list,
    k: int,
    figures_dir: str = None,
    plot_func: Callable = None
) -> Dict[str, np.ndarray]:
    """
    Run RFE with all configured classifier models.
    
    Args:
        X_normalized: Normalized feature matrix
        y: Target labels
        columns: Original feature column names
        k: Number of features to select
        figures_dir: Directory to save rank plots
        plot_func: Optional plotting function
    
    Returns:
        Dictionary mapping "RFE_<model_name>" to selected feature indices
    """
    models = get_rfe_models()
    results = {}
    
    for model_name, model in models.items():
        indices = rfe_select(
            X_normalized, y, columns, model, model_name, k, figures_dir, plot_func
        )
        results[f'RFE_{model_name}'] = indices
    
    return results
