#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCM-DL Talent Identification using Machine Learning

Clean, modular refactoring of the original implementation.
Maintains identical accuracy (94-97%) on TID dataset.

Author: Muhammet Gökhan Erdem
License: ComVIS Lab
Contact: merdem@comvislab.com
Website: https://comvislab.com
GitHub: https://github.com/Comvislab/TID
REF: https://dx.doi.org/10.1109/ACCESS.2025.3562551

Usage:
    python main.py <num_features_to_select>
    
Example:
    python main.py 6
"""

import os
import sys
import pandas as pd

# Disable GPU if not needed
from config import USE_GPU
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import configuration
from config import (
    SEED, DATASET_DIR, DATASET_FILE, TEST_SIZE, RANDOM_STATE,
    LASSO_ALPHA, SCM_DL_CONFIG
)

# Import utilities
from utils.seed import set_seed
from utils.io_helpers import get_timestamp, create_results_dirs

# Import data handling
from data.loader import load_dataset, split_data, save_splits
from data.preprocessing import normalize_features

# Import feature selection methods
from feature_selection.statistical import select_kbest_all
from feature_selection.lasso_selector import lasso_select
from feature_selection.boruta_selector import boruta_select
from feature_selection.rfe_selector import rfe_select_all

# Import evaluation
from evaluation.visualization import plot_feature_ranks
from evaluation.metrics import confusion_analysis

# Import classifiers
from classifiers.factory import get_sklearn_classifiers, create_scm_dl_model


def parse_arguments():
    """Parse command line arguments."""
    if len(sys.argv) < 2:
        print("Command Should be like:")
        print(">>> python main.py NUM_of_SELECTED_FEATURES")
        sys.exit(1)
    
    num_features = int(sys.argv[1])
    print(f'The Selected Feature Count: {num_features}')
    return num_features


def run_feature_selection(X_normalized, y_train, columns, k, figures_dir):
    """
    Run all feature selection methods.
    
    Args:
        X_normalized: Normalized training features
        y_train: Training labels
        columns: Feature column names
        k: Number of features to select
        figures_dir: Directory to save figures
    
    Returns:
        Dictionary mapping method names to selected feature indices
    """
    selected_features = {}
    
    # SelectKBest methods (chi2, ANOVA, MI)
    print("\n" + "="*60)
    print("Running SelectKBest feature selection...")
    print("="*60)
    kbest_results = select_kbest_all(
        X_normalized, y_train, columns, k, figures_dir, plot_feature_ranks
    )
    selected_features.update(kbest_results)
    
    # Lasso-based selection
    print("\n" + "="*60)
    print("Running Lasso feature selection...")
    print("="*60)
    lasso_indices = lasso_select(
        X_normalized, y_train, columns, k, LASSO_ALPHA, figures_dir, plot_feature_ranks
    )
    selected_features['LassoKBest'] = lasso_indices
    
    # Boruta selection
    print("\n" + "="*60)
    print("Running Boruta feature selection...")
    print("="*60)
    boruta_indices = boruta_select(
        X_normalized, y_train, columns, k, figures_dir, plot_feature_ranks
    )
    selected_features['BorutaKBest_RF'] = boruta_indices
    
    # RFE with multiple classifiers
    print("\n" + "="*60)
    print("Running RFE feature selection...")
    print("="*60)
    rfe_results = rfe_select_all(
        X_normalized, y_train, columns, k, figures_dir, plot_feature_ranks
    )
    selected_features.update(rfe_results)
    
    return selected_features


def compute_feature_set_stats(selected_features, results_file):
    """
    Compute shared and union of all selected feature sets.
    
    Args:
        selected_features: Dictionary of method -> indices
        results_file: Path to results file
    """
    list_shared = None
    list_union = None
    
    with open(results_file, "w") as f:
        f.write("===== SELECTED FEATURES with THEIR SELECTION METHOD ======\n")
    
    for key, indices in selected_features.items():
        if list_shared is None:
            list_shared = set(indices)
            list_union = set(indices)
        else:
            list_shared = list_shared.intersection(set(indices))
            list_union = list_union.union(set(indices))
        
        with open(results_file, "a") as f:
            f.write(f"Selected Features of {key}: {list(indices)}\n")
        
        print(f"{key}: {indices}")
    
    print(f"Ortak Feature Set: {list(list_shared)}")
    
    with open(results_file, "a") as f:
        f.write(f"Ortak   Feature Set: {list(list_shared)}\n")
        f.write(f"Bileske Feature Set: {list(list_union)}\n")


def train_and_evaluate_classifiers(
    X_train, X_test, y_train, y_test,
    selected_features, num_features,
    results_file, confusions_dir
):
    """
    Train and evaluate all classifiers with each feature selection method.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        selected_features: Dictionary of method -> feature indices
        num_features: Number of selected features
        results_file: Path to results file
        confusions_dir: Directory for confusion matrices
    
    Returns:
        Dictionary of results for DataFrame creation
    """
    # Get sklearn classifiers
    classifiers = get_sklearn_classifiers()
    
    # Add SCM_DL model
    scm_dl_model = create_scm_dl_model(num_features)
    classifiers.append((scm_dl_model, 'SCM_DL'))
    
    result_dict = {}
    
    with open(results_file, "a") as f:
        f.write("Classifier:        FeatureSelector:      Accuracy:\n")
    
    for model, model_name in classifiers:
        print(f"\n{'-'*60}")
        print(f"CLASSIFIER: {model_name}")
        print(f"Selected Features Count: {num_features}")
        print(f"{'-'*60}")
        
        for method_name, feature_indices in selected_features.items():
            print(f"Classifier: {model_name} | Feature Selection Method: {method_name}")
            
            # Check if this is a deep learning model
            is_dl_model = hasattr(model, 'deepname') and model_name in ['SCM_DL', 'ShallowDL']
            
            if is_dl_model:
                # Train deep learning model
                model.train(
                    X_train[:, feature_indices], 
                    y_train,
                    model.epochs,
                    model.batch_size,
                    model.validation_split
                )
                loss, accuracy = model.evaluate(X_test[:, feature_indices], y_test)
                y_pred = model.predict(X_test[:, feature_indices])
            else:
                # Train sklearn model
                model.fit(X_train[:, feature_indices], y_train)
                accuracy = model.score(X_test[:, feature_indices], y_test)
                y_pred = model.predict(X_test[:, feature_indices])
            
            # Confusion matrix analysis
            confusion_analysis(
                y_test, y_pred, 
                f"{model_name}_{method_name}",
                confusions_dir,
                is_dl_model=is_dl_model
            )
            
            print(f"Accuracy on Test: {accuracy}\n")
            
            # Save results
            with open(results_file, "a") as f:
                f.write(f"Classifier: {model_name}  FeatureSelector: {method_name},  Accuracy: {accuracy}\n")
            
            if model_name not in result_dict:
                result_dict[model_name] = []
            result_dict[model_name].append(accuracy)
    
    return result_dict


def save_results_dataframe(result_dict, selected_features, results_path, num_features, timestamp):
    """
    Save results as a DataFrame CSV file.
    
    Args:
        result_dict: Dictionary of classifier -> accuracy list
        selected_features: Dictionary of feature selection methods
        results_path: Path to results directory
        num_features: Number of selected features
        timestamp: Timestamp string
    """
    result_df = pd.DataFrame(result_dict)
    result_df.insert(0, "FeatureSelector", list(selected_features.keys()), True)
    
    output_path = os.path.join(
        results_path, 
        f"ResultDF_{num_features}_Features_{timestamp}.csv"
    )
    result_df.to_csv(output_path, sep=";")
    print(f"\nResults saved to: {output_path}")


def main():
    """Main execution function."""
    # Parse arguments
    num_features = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed(SEED)
    
    # Generate timestamp
    timestamp = get_timestamp()
    print(f"Timestamp: {timestamp}")
    
    # Create output directories
    results_path, figures_dir, confusions_dir, results_file = create_results_dirs(
        num_features, timestamp
    )
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    X, y, data = load_dataset(DATASET_DIR, DATASET_FILE)
    columns = list(data.drop(["Branch"], axis=1).columns)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    
    # Save train/test splits
    save_splits(X_train, X_test, y_train, y_test, data.columns, DATASET_DIR, timestamp)
    
    # Normalize features
    X_normalized, scaler = normalize_features(X_train)
    
    # Run feature selection
    selected_features = run_feature_selection(
        X_normalized, y_train, columns, num_features, figures_dir
    )
    
    # Compute shared/union feature sets and save
    compute_feature_set_stats(selected_features, results_file)
    
    # Train and evaluate classifiers
    result_dict = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test,
        selected_features, num_features,
        results_file, confusions_dir
    )
    
    # Save results as DataFrame
    save_results_dataframe(
        result_dict, selected_features, results_path, num_features, timestamp
    )
    
    print("\n" + "="*60)
    print("Execution completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
