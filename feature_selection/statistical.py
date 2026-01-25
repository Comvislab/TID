# -*- coding: utf-8 -*-
"""
Statistical Feature Selection Methods

SelectKBest with chi2, ANOVA (f_classif), and Mutual Information.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from typing import Dict, Callable


def select_kbest(
    X_normalized: np.ndarray,
    y: np.ndarray,
    columns: list,
    method: Callable,
    k: int,
    figures_dir: str,
    plot_func: Callable = None
) -> np.ndarray:
    """
    Select top-k features using SelectKBest with specified scoring method.
    
    Args:
        X_normalized: Normalized feature matrix (required for chi2)
        y: Target labels
        columns: Original feature column names
        method: Scoring function (chi2, f_classif, or mutual_info_classif)
        k: Number of features to select
        figures_dir: Directory to save rank plots
        plot_func: Optional plotting function
    
    Returns:
        Array of selected feature indices
    """
    # Get method name for display
    method_names = {
        chi2: "chi2",
        f_classif: "ANOVA",
        mutual_info_classif: "MI"
    }
    method_name = method_names.get(method, str(method))
    
    # Fit SelectKBest
    selector = SelectKBest(score_func=method, k=k)
    selector.fit(X_normalized, y)
    
    # Create scores DataFrame
    df_scores = pd.DataFrame(selector.scores_)
    df_columns = pd.DataFrame(columns)
    
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = [f'SelKBest Specs{method_name}', f'SelKBestScore{method_name}']
    
    # Get top-k features
    best_features = feature_scores.nlargest(k, f'SelKBestScore{method_name}')
    selected_indices = best_features.index.values
    
    print(f"SelectKBest {method_name} (n_largest are selected)", selector.scores_)
    
    # Plot if function provided
    if plot_func is not None:
        plot_func(f"SelectKBest {method_name} (n_largest are selected)", selector.scores_, figures_dir)
    
    return selected_indices


def select_kbest_all(
    X_normalized: np.ndarray,
    y: np.ndarray,
    columns: list,
    k: int,
    figures_dir: str,
    plot_func: Callable = None
) -> Dict[str, np.ndarray]:
    """
    Run SelectKBest with all three methods: chi2, ANOVA, and Mutual Information.
    
    Args:
        X_normalized: Normalized feature matrix
        y: Target labels
        columns: Original feature column names
        k: Number of features to select
        figures_dir: Directory to save rank plots
        plot_func: Optional plotting function
    
    Returns:
        Dictionary mapping method names to selected feature indices
    """
    methods = {
        'SelKBest_chi2': chi2,
        'SelKBest_ANOVA': f_classif,
        'SelKBest_MI': mutual_info_classif
    }
    
    results = {}
    for name, method in methods.items():
        indices = select_kbest(X_normalized, y, columns, method, k, figures_dir, plot_func)
        results[name] = indices
    
    return results
