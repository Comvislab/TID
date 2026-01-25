# -*- coding: utf-8 -*-
"""
Visualization Functions

Plotting for feature ranks, class distributions, and other visualizations.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_ranks(
    method_name: str,
    ranks: list,
    figures_dir: str
) -> None:
    """
    Visualize feature ranking as a bar chart.
    
    Args:
        method_name: Name of the feature selection method
        ranks: List of feature rankings or scores
        figures_dir: Directory to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.title(method_name)
    plt.xlabel("Feature Index")
    plt.ylabel("Ranking")
    plt.bar(range(len(ranks)), ranks)
    plt.savefig(os.path.join(figures_dir, method_name))
    plt.close()


def plot_class_distribution(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str = 'Branch'
) -> None:
    """
    Visualize class distribution in the dataset.
    
    Args:
        df: DataFrame containing the data
        dataset_name: Name of the dataset for plot title
        target_column: Name of the target column (default: 'Branch')
    """
    # Get unique classes
    unique_classes = df[target_column].unique()
    print(f"Unique classes: {unique_classes}")
    
    # Count occurrences
    class_distribution = df[target_column].value_counts()
    
    # Plot
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind='bar', color='skyblue')
    plt.title(f'Class Distribution of {dataset_name}')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
