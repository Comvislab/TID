# -*- coding: utf-8 -*-
"""
Classifier Factory

Creates and configures all classifier models including sklearn and deep learning.
"""

import os
import sys
from typing import Dict, Any, List, Tuple

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from SCM_DL_BaseModel import SCM_DL_Base, ShallowDL

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RFE_CLASSIFIERS, SCM_DL_CONFIG, SHALLOW_DL_CONFIG


def get_sklearn_classifiers() -> List[Tuple[Any, str]]:
    """
    Create sklearn classifiers for evaluation.
    
    These are the same classifiers used in RFE, but fresh instances
    for the final evaluation phase.
    
    Returns:
        List of (model, name) tuples
    """
    classifiers = [
        (RandomForestClassifier(**RFE_CLASSIFIERS['RFC']), 'RFC'),
        (ExtraTreesClassifier(**RFE_CLASSIFIERS['ETC']), 'ETC'),
        (SVC(**RFE_CLASSIFIERS['SVC']), 'SVC'),
        (DecisionTreeClassifier(**RFE_CLASSIFIERS['DTC']), 'DTC'),
    ]
    return classifiers


def create_scm_dl_model(input_dim: int) -> SCM_DL_Base:
    """
    Create and build the SCM_DL_Base model with configured parameters.
    
    Args:
        input_dim: Number of input features (selected features count)
    
    Returns:
        Compiled SCM_DL_Base model ready for training
    """
    model = SCM_DL_Base(
        layer_units=SCM_DL_CONFIG['layer_units'],
        activations=SCM_DL_CONFIG['activations'],
        input_dim=input_dim,
        output_dim=SCM_DL_CONFIG['output_dim'],
        Optmzr=SCM_DL_CONFIG['optimizer'],
        Lossf=SCM_DL_CONFIG['loss'],
        epochs=SCM_DL_CONFIG['epochs'],
        batch_size=SCM_DL_CONFIG['batch_size'],
        validation_split=SCM_DL_CONFIG['validation_split'],
        learning_rate=SCM_DL_CONFIG['learning_rate'],
        deepname='SCM_DL_Base'
    )
    model.build_model()
    model.compile_model(model.Optmzr, model.learning_rate, model.Lossf)
    model.summary()
    
    return model


def create_shallow_dl_model(input_dim: int) -> ShallowDL:
    """
    Create and build the ShallowDL model with configured parameters.
    
    Args:
        input_dim: Number of input features (selected features count)
    
    Returns:
        Compiled ShallowDL model ready for training
    """
    model = ShallowDL(
        layer_units=SHALLOW_DL_CONFIG['layer_units'],
        activations=SHALLOW_DL_CONFIG['activations'],
        input_dim=input_dim,
        output_dim=SHALLOW_DL_CONFIG['output_dim'],
        Optmzr=SHALLOW_DL_CONFIG['optimizer'],
        Lossf=SHALLOW_DL_CONFIG['loss'],
        epochs=SHALLOW_DL_CONFIG['epochs'],
        batch_size=SHALLOW_DL_CONFIG['batch_size'],
        validation_split=SHALLOW_DL_CONFIG['validation_split'],
        learning_rate=SHALLOW_DL_CONFIG['learning_rate'],
        deepname='ShallowDL'
    )
    model.build_model()
    model.compile_model(model.Optmzr, model.learning_rate, model.Lossf)
    model.summary()
    
    return model


def get_all_classifiers(input_dim: int) -> List[Tuple[Any, str]]:
    """
    Get all classifiers including sklearn and deep learning models.
    
    Args:
        input_dim: Number of input features for DL models
    
    Returns:
        List of (model, name) tuples
    """
    classifiers = get_sklearn_classifiers()
    classifiers.append((create_scm_dl_model(input_dim), 'SCM_DL'))
    
    return classifiers
