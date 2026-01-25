# -*- coding: utf-8 -*-
"""
Evaluation Metrics

Confusion matrix analysis, sensitivity, specificity, and ROC approximation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import ROC calculation from models
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from CinsConMat import CinsRocRecPrec


def confusion_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_and_feature: str,
    confusions_dir: str,
    is_dl_model: bool = False
) -> None:
    """
    Compute and visualize confusion matrix with sensitivity and specificity.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels or probabilities
        model_and_feature: String identifier for the model and feature combo
        confusions_dir: Directory to save confusion matrix plots
        is_dl_model: Whether predictions are from a DL model (need argmax)
    """
    print(f"Confusion_Analysis is starting for {model_and_feature}")
    
    # Convert probabilities to class labels for DL models
    if is_dl_model:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate sensitivity (recall) for each class
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    
    # Calculate specificity for each class
    specificity = []
    for i in range(cm.shape[0]):
        true_negatives = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        spec = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity.append(spec)
    
    # Print metrics for each class
    for i in range(len(sensitivity)):
        print(f"Class {i}: Sensitivity = {sensitivity[i]}, Specificity = {specificity[i]}")
    
    # Save confusion matrix as CSV
    np.savetxt(
        os.path.join(confusions_dir, f"{model_and_feature}.csv"), 
        cm, 
        fmt='%d;'
    )
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title(f'Confusion Matrix of {model_and_feature}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(confusions_dir, model_and_feature))
    plt.close()
    
    # Calculate approximate ROC and save figure
    roc_fig = CinsRocRecPrec(cm)
    roc_fig.savefig(os.path.join(confusions_dir, f"ROC_{model_and_feature}"))
    plt.close(roc_fig)
