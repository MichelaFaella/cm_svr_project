# Support Vector Regression with Nesterov Smoothing algorithm

<p align = "center">
  <img src = "https://upload.wikimedia.org/wikipedia/it/e/e2/Stemma_unipi.svg" width="256" height="256">
</p>

<p align = "center">
  Support Vector Regression built from scratch using Nesterov's smoothed gradient method.  
  A project for  
  Computational Mathematics for Learning and Data Analysis  
  course of Computer Science - Artificial Intelligence at University of Pisa.
</p>

---

## Project Description and Introduction

This project implements **Support Vector Regression (SVR)** from scratch, based on Nesterov's smoothed dual optimization technique.  
It includes a full machine learning pipeline:

- Custom SVR with support for RBF, Polynomial, and Linear kernels
- Smoothed gradient method from Nesterov (2005)
- Outlier removal
- Data normalization (Z-score and MinMax)
- Hyperparameter tuning (Grid and Random Search)
- Training/validation/testing split
- Result plotting and metrics

---

## Authors

- **Michela Faella** - *Developer* - (https://github.com/MichelaFaella)
- **Margherita Merialdo** - *Developer* - (https://github.com/margheritamerialdo)

---
## 📁 Repository Structure

```text
.
├── dataset_diamonds/
│   └── diamonds_cleaned.csv        # Dataset used for regression
├── SVM/
│   ├── Svr.py                      # Custom SVR implementation (Nesterov)
│   ├── Solver.py                   # CVXPY-based dual solver
│   ├── Kernels.py                  # Linear, Polynomial, RBF kernels
│   ├── Search.py                   # Grid & random search for hyperparameters
│   ├── Enum.py                     # KernelType enum
│   └── utility.py                  # Preprocessing, metrics, plotting
├── main_d.py                       # Main for Nesterov smoothed SVR
├── main_CVXPY.py                   # Main for CVXPY dual solver
└── README.md                       # This file
```

---

## Documentation

The project is organized in the following Python modules:

- `Svr.py` – Core SVR class with training and prediction logic
- `Kernels.py` – Implementation of Linear, RBF, and Polynomial kernels
- `Enum.py` – Enum for kernel type selection
- `utility.py` – Preprocessing, normalization, outlier handling, reporting
- `Search.py` – Grid search and random search for hyperparameter tuning
- `Solver.py` – Output capturing class for stdout redirection

---

## Technical Information

### Requirements

- Python 3.9+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`


