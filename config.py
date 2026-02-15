# -*- coding: utf-8 -*-
"""
Configuration and Hyperparameters for SCM-DL Talent Identification

All hyperparameters are preserved from the original implementation
to maintain 94-97% accuracy on TID dataset.

Author: Muhammet Gökhan Erdem
"""

# =============================================================================
# Random Seed
# =============================================================================
SEED        = 42
NUM_CLASSES = 6
EPOCHS      = 200
BATCH_SIZE  = 32

# =============================================================================
# Paths
# =============================================================================
DATASET_DIR = 'DataSetTID'
# DATASET_FILE = 'TIDMLStage2_numericcateg_240404.csv'
# Alternative datasets:
DATASET_FILE = 'TIDMLStage2_numericcateg_10K.csv'
# DATASET_FILE = 'TIDMLStage2_numericcateg_100K.csv'

# =============================================================================
# Data Splitting
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# SCM_DL Model Hyperparameters (Critical for accuracy!)
# =============================================================================
SCM_DL_CONFIG = {
    'layer_units': [[128, 128, 64], [128, 128, 64], [128, 128, 64]],
    'activations': [
        ['sigmoid', 'sigmoid', 'sigmoid'],
        ['sigmoid', 'sigmoid', 'sigmoid'],
        ['sigmoid', 'sigmoid', 'sigmoid']
    ],
    'output_dim': NUM_CLASSES,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'validation_split': 0.1,
    'learning_rate': 0.00001,
}

# =============================================================================
# ShallowDL Model Hyperparameters
# =============================================================================
SHALLOW_DL_CONFIG = {
    'layer_units': [128, 128, 64],
    'activations': ['sigmoid', 'sigmoid', 'sigmoid'],
    'output_dim': NUM_CLASSES,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'validation_split': 0.1,
    'learning_rate': 0.00001,
}

# =============================================================================
# Feature Selection Hyperparameters
# =============================================================================
# Lasso
LASSO_ALPHA = 0.01

# Boruta (with Random Forest)
BORUTA_RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 7,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'random_state': SEED,
}

# =============================================================================
# RFE Classifier Configurations
# =============================================================================
RFE_CLASSIFIERS = {
    'RFC': {
        'n_estimators': 100,
        'max_depth': 7,
        'random_state': SEED,
    },
    'ETC': {
        'random_state': SEED,
    },
    'SVC': {
        'kernel': 'linear',
        'C': 10,
        'random_state': SEED,
    },
    'DTC': {
        'max_depth': 7,
        'random_state': SEED,
    },
}

# =============================================================================
# GPU Settings
# =============================================================================
USE_GPU = False  # Set to True to enable GPU
