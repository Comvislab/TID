# SCM-DL: Talent Identification using Machine Learning

A modular Python implementation for talent identification using Supply Chain Management combined with Deep Learning (SCM-DL).

## 📋 Overview

This project implements a machine learning pipeline for talent identification that achieves **94-97% accuracy** on the TID dataset. The codebase has been refactored from a monolithic script into a clean, modular package structure.

## 🏗️ Project Structure

```
SRC/
├── main.py                     # Entry point - orchestrates the entire pipeline
├── config.py                   # Centralized configuration and hyperparameters
├── utils/
│   ├── seed.py                 # Random seed management for reproducibility
│   └── io_helpers.py           # File/directory utilities
├── data/
│   ├── loader.py               # Dataset loading and train/test splitting
│   └── preprocessing.py        # Feature normalization (MinMaxScaler)
├── feature_selection/
│   ├── statistical.py          # SelectKBest (chi2, ANOVA, Mutual Info)
│   ├── lasso_selector.py       # Lasso regression-based selection
│   ├── boruta_selector.py      # Boruta with Random Forest
│   └── rfe_selector.py         # RFE with RFC, ETC, SVC, DTC
├── evaluation/
│   ├── metrics.py              # Confusion matrix, sensitivity, specificity
│   └── visualization.py        # Feature ranking plots, class distribution
├── classifiers/
│   └── factory.py              # Model creation (sklearn + SCM_DL)
└── models/
    ├── SCM_DL_BaseModel.py     # Custom deep learning architecture
    └── CinsConMat.py           # ROC and precision/recall calculations
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn boruta
```

### Running the Pipeline

```bash
cd SRC
python main.py <num_features_to_select>
```

**Example:**
```bash
python main.py 6
```

This runs the complete pipeline:
1. Loads the TID dataset
2. Applies 8 feature selection methods
3. Trains 5 classifiers on each feature subset
4. Saves results, confusion matrices, and ROC plots

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEED` | 42 | Random seed for reproducibility |
| `TEST_SIZE` | 0.2 | Train/test split ratio |
| `epochs` | 200 | Training epochs for DL models |
| `batch_size` | 16 | Batch size for training |
| `learning_rate` | 0.00001 | Learning rate for Adam optimizer |
| `layer_units` | [[128,128,64]×3] | SCM_DL network architecture |

## 📊 Feature Selection Methods

| Method | Type | Description |
|--------|------|-------------|
| SelectKBest (chi2) | Statistical | Chi-squared test |
| SelectKBest (ANOVA) | Statistical | F-value (ANOVA) |
| SelectKBest (MI) | Statistical | Mutual Information |
| Lasso | Regularization | L1 penalty coefficients |
| Boruta | Ensemble | Random Forest wrapper |
| RFE-RFC | Recursive | Random Forest Classifier |
| RFE-ETC | Recursive | Extra Trees Classifier |
| RFE-SVC | Recursive | Support Vector Classifier |
| RFE-DTC | Recursive | Decision Tree Classifier |

## 🤖 Classifiers

- **RFC** - Random Forest Classifier
- **ETC** - Extra Trees Classifier
- **SVC** - Support Vector Classifier (linear kernel)
- **DTC** - Decision Tree Classifier
- **SCM_DL** - Custom multi-branch deep learning model

## 📁 Output Structure

After running, results are saved to `Results_<num_features>/`:

```
Results_6/
├── Figures26_01_25_1640/          # Feature ranking plots
├── Confusions26_01_25_1640/       # Confusion matrices and ROC curves
├── FE_Results_26_01_25_1640.txt   # Detailed results log
└── ResultDF_6_Features_*.csv      # Summary DataFrame
```

## 📈 Expected Accuracy

With 6 selected features on the TID dataset:
- **SCM_DL**: 94-97%
- **RFC**: ~90-93%
- **ETC**: ~88-92%

## 📚 References

- **Paper**: [https://dx.doi.org/10.1109/ACCESS.2025.3562551](https://dx.doi.org/10.1109/ACCESS.2025.3562551)
- **GitHub**: [https://github.com/Comvislab/TID](https://github.com/Comvislab/TID)

## Citation
If you use our work, please cite our paper:

```
- @ARTICLE{10971350,
  author={Abidin, Didem and Erdem, Muhammed G.},
  journal={IEEE Access}, 
  title={SCM-DL: Split-Combine-Merge Deep Learning Model Integrated With Feature Selection in Sports for Talent Identification}, 
  year={2025},
  volume={13},
  number={},
  pages={71148-71172},
  keywords={Sports;Adaptation models;Feature extraction;Deep learning;Artificial neural networks;Data models;Psychology;Object recognition;Motors;Random forests;Deep learning;neural networks;recursive feature selection;talent identification},
  doi={10.1109/ACCESS.2025.3562551}}
```

## 👤 Author

**Muhammet Gökhan Erdem**
- Email: merdem@comvislab.com
- Website: [https://comvislab.com](https://comvislab.com)

## 📄 License

ComVIS Lab - All rights reserved.
