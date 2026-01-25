# -*- coding: utf-8 -*-
"""
I/O Helper Functions

Directory creation, timestamp generation, and file utilities.
"""

import os
import datetime
from typing import Tuple


def get_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Formatted timestamp string (YY_MM_DD_HHMM)
    """
    now = datetime.datetime.now()
    return now.strftime("%y_%m_%d_%H%M")


def create_results_dirs(num_features: int, timestamp: str) -> Tuple[str, str, str, str]:
    """
    Create the results directory structure.
    
    Args:
        num_features: Number of selected features (used in folder name)
        timestamp: Timestamp string for unique folder names
    
    Returns:
        Tuple of (results_path, figures_dir, confusions_dir, results_file)
    """
    # Main results directory
    results_path = f'./Results_{num_features}'
    
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        print(f"Folder {results_path} created!")
    else:
        print(f"Folder {results_path} already exists")
    
    # Subdirectories for figures and confusion matrices
    figures_dir = os.path.join(results_path, f"Figures{timestamp}")
    confusions_dir = os.path.join(results_path, f"Confusions{timestamp}")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(confusions_dir, exist_ok=True)
    
    # Results text file
    results_file = os.path.join(results_path, f"FE_Results_{timestamp}.txt")
    
    # Initialize results file
    with open(results_file, "w") as f:
        f.write(f" ************ FEATURE ELIMINATIONs and RESULTS {timestamp} *********:\n")
    
    return results_path, figures_dir, confusions_dir, results_file


def only_upper(s: str) -> str:
    """
    Extract only uppercase characters from a string.
    
    Args:
        s: Input string
    
    Returns:
        String containing only uppercase characters
    """
    return "".join(c for c in s if c.isupper())
