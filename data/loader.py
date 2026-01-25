# -*- coding: utf-8 -*-
"""
Dataset Loading and Splitting

Handles loading CSV data, splitting into train/test sets, and saving splits.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_dataset(dataset_dir: str, dataset_file: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the TID dataset from CSV file.
    
    Args:
        dataset_dir: Directory containing the dataset
        dataset_file: Name of the CSV file
    
    Returns:
        Tuple of (X features array, y labels array, original DataFrame)
    """
    filepath = os.path.join(dataset_dir, dataset_file)
    data = pd.read_csv(filepath)
    
    X = data.drop(["Branch"], axis=1).values
    y = data["Branch"].values
    
    return X, y, data


def split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Label vector
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_splits(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    columns: list,
    dataset_dir: str,
    timestamp: str
) -> None:
    """
    Save train and test splits as CSV files.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        columns: Column names from original DataFrame
        dataset_dir: Directory to save CSV files
        timestamp: Timestamp for filename
    """
    # Create and save training DataFrame
    df_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
    df_train.columns = columns
    df_train.to_csv(os.path.join(dataset_dir, f'CinsTrain_{timestamp}.csv'), index=False)
    
    # Create and save testing DataFrame
    df_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
    df_test.columns = columns
    df_test.to_csv(os.path.join(dataset_dir, f'CinsTest_{timestamp}.csv'), index=False)
