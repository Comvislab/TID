# -*- coding: utf-8 -*-
"""
Random Seed Management

Sets seeds for reproducibility across all random number generators.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    This ensures consistent results across runs by setting seeds for:
    - Python's random module
    - NumPy random generator
    - TensorFlow random generator
    - Python hash seed
    
    Args:
        seed: The random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
