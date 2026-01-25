# -*- coding: utf-8 -*-
"""
Lasso-based Feature Selection

Uses Lasso regression coefficients to rank and select features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from typing import Callable


def lasso_select(
    X_normalized: np.ndarray,
    y: np.ndarray,
    columns: list,
    k: int,
    alpha: float = 0.01,
    figures_dir: str = None,
    plot_func: Callable = None
) -> np.ndarray:
    """
    Select top-k features using Lasso regression coefficients.
    
    Features with larger absolute coefficients are considered more important.
    
    Args:
        X_normalized: Normalized feature matrix
        y: Target labels
        columns: Original feature column names
        k: Number of features to select
        alpha: Lasso regularization parameter (default: 0.01)
        figures_dir: Directory to save rank plots
        plot_func: Optional plotting function
    
    Returns:
        Array of selected feature indices
    """
    # Fit Lasso
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_normalized, y)
    
    # Create scores DataFrame using absolute coefficients
    lasso_scores = pd.DataFrame(np.abs(lasso.coef_))
    lasso_columns = pd.DataFrame(columns)
    
    feature_scores = pd.concat([lasso_columns, lasso_scores], axis=1)
    feature_scores.columns = ['Lasso_Specs', 'Lasso_Scores']
    
    # Get top-k features by score
    best_features = feature_scores.nlargest(k, 'Lasso_Scores')
    selected_indices = best_features.index.values
    
    # Plot if function provided
    if plot_func is not None and figures_dir is not None:
        plot_func("Lasso_KBest (n_largest are selected)", np.abs(lasso.coef_), figures_dir)
    
    return selected_indices
