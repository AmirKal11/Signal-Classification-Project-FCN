# Di-Tau Signal Classification using Deep Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project implements a **Fully Connected Neural Network (DNN)** to classify high-energy physics events from the ATLAS detector. The goal is to distinguish rare **Higgs boson decays** ($H \to \tau\tau$, Signal) from common background noise ($Z \to \tau\tau$, Background).

This is a classic **imbalanced binary classification problem** involving high-dimensional kinematic data.

**Key Achievements:**
* Achieved **AUC-ROC Score of 0.95**.
* Implemented a robust **ETL pipeline** handling H5 file structures and dynamic feature extraction.
* Designed a **Physics Integrity Check** system to catch "silent failures" (e.g., dead sensors/columns) before training.
* Applied domain-specific **Kinematic Filtering** to reduce noise by ~60%.

## Results
The model was evaluated on a hold-out test set of ~40,000 events.

| Metric | Score |
| :--- | :--- |
| **AUC** | **0.93** |
| **Accuracy** | 0.88 |
| **Precision** | 0.74 |
| **Recall** | 0.88 |

*(Note: High recall was prioritized to ensure no potential signal events were missed.)*

![ROC Curve](results/roc_curve.png)

## Methodology

### 1. Data Engineering (`src/data_loader.py`)
* **Robust Loading:** Custom `H5DataLoader` class handles inconsistent column naming and shape mismatches (1D vs 2D arrays) across large H5 datasets.
* **Merging:** Dynamically merges event-level metadata with particle-level kinematic data (Jets, Leptons, MET).
* **Mock Compatibility:** Includes aliasing logic to seamlessly handle both real ATLAS ntuples and generated mock data.

### 2. Physics Preprocessing (`src/logic_preprocessing.py`)
* **Integrity Guardrails:** Implements a "Dead Column Check" that halts execution if feature columns are empty or zero-variance, preventing silent model failure.
* **Event Topology Filtration:** Applies strict physics cuts (e.g., $p_T > 27$ GeV, $\ge 3$ Jets) to ensure data quality.
* **Unit Standardization:** Auto-corrects energy units (MeV to GeV) and applies Standard Scaling for numerical stability.

### 3. Exploratory Data Analysis (`src/visualization.py`)
* **Physics Verification:** Includes a modular `PhysicsVisualizer` class to generate comparative histograms (Signal vs. Background).
* **Unit-Aware Plotting:** Automatically detects high-dynamic-range variables (e.g., $p_T$, MET) and applies Log-Scaling to visualize "long tail" distributions.
* **Correlation Analysis:** Generates heatmaps to identify multicollinearity between kinematic variables (e.g., MET vs. SumET).

### 4. Model Architecture (`src/model.py`)
* **Architecture:** Fully Connected Deep Neural Network (DNN).
* **Specs:** 6 Hidden Layers, PReLU activation, and Dropout (0.2) for regularization.
* **Training:** Optimized using Adam optimizer and Binary Crossentropy loss with **Class Weights** to handle signal imbalance.

## Repository Structure
```text
├── data/                   # Raw H5 datasets (Signal/Background)
├── src/
│   ├── data_loader.py      # H5 parsing and dataframe merging
│   ├── logic_preprocessing.py # Kinematic cuts, scaling, and integrity checks
│   ├── visualization.py    # EDA, physics plots, and learning curves
│   ├── model.py            # Keras model architecture
│   └── utils.py            # Helper functions
├── main.ipynb              # Executive notebook (Run this to reproduce results)
├── requirements.txt        # Dependencies
├── README.md               # Project Documentation
└── Physics background.pdf  # Explaining the physics behind the project