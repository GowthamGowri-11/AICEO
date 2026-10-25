# Artificial Neural Network (ANN) for Binary Classification Using the make_moons Dataset

A simple, efficient, and academically sound machine learning project demonstrating binary classification on non-linearly separable data using an Artificial Neural Network (ANN) built with TensorFlow/Keras and Scikit-learn's `make_moons` dataset.

---

## 1. Project Objective
The objective of this project is to develop and train an Artificial Neural Network (ANN) that learns to classify 2D data points into two distinct classes (Class 0 and Class 1). Because the `make_moons` dataset is non-linearly separable, linear models fail. This project showcases how an ANN with non-linear activation functions (ReLU and Sigmoid), paired with validation-based early stopping, reliably learns the curved decision boundary.

---

## 2. Technology Stack
Only essential libraries are used:
- **Python** (Core programming language)
- **NumPy** (Numerical computing and array operations)
- **Scikit-learn** (Dataset generation & train-test split)
- **TensorFlow / Keras** (Neural network modeling, training & evaluation)
- **Matplotlib** (Data and decision boundary visualization)

---

## 3. Project Structure
```text
ANN-PROJECT/
├── main.py                  # Complete implementation: data, training, evaluation, visualization
├── requirements.txt         # Required Python dependencies
├── README.md                # Project documentation and instructions
└── results_visualization.png # Generated high-resolution output plots
```

---

## 4. Neural Network Architecture & Training Strategy

| Layer | Type | Neurons | Activation | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Input Layer | 2 | - | Accepts 2D coordinates $(X_1, X_2)$ |
| **Hidden 1** | Dense | 32 | ReLU | Non-linear geometric feature extraction |
| **Hidden 2** | Dense | 16 | ReLU | Hierarchical feature refinement |
| **Output** | Dense | 1 | Sigmoid | Outputs probability $P(Y=1)$ between 0 and 1 |

### Compilation & Hyperparameters:
- **Optimizer:** Adam (tuned learning rate $\alpha = 0.005$ for efficient convergence)
- **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
- **Metric:** Accuracy
- **Batch Size:** 32
- **Validation Split:** 20% validation split on training data
- **Early Stopping:** Monitored on `val_loss` with patience=20 and `restore_best_weights=True`

---

## 5. Performance Comparison

| Metric | Original Baseline Model | Improved Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Architecture** | 16 → 8 → 1 | 32 → 16 → 1 | Balanced capacity |
| **Learning Rate** | 0.001 (default) | 0.005 | Accelerated convergence |
| **Regularization** | None (fixed 100 epochs) | Early Stopping (patience=20) | Prevents overfitting |
| **Test Accuracy** | **91.00%** | **98.00%** | **+7.00%** |
| **Test Loss** | **0.1814** | **0.0608** | **-66.5% reduction** |
| **Misclassified Test Samples** | 9 / 100 | **2 / 100** | **7 fewer errors** |

---

## 6. Installation & Execution Steps

### Prerequisites
Make sure Python 3.9+ is installed on your machine.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Program
```bash
python main.py
```

---

## 7. Visual Outputs
When executed, a single 3-panel figure is displayed and saved to `results_visualization.png`:
1. **Dataset Visualization:** Scatter plot showing the two moon-shaped interleaving classes.
2. **Training & Validation Performance:** Loss and accuracy curves across epochs for both training and validation sets.
3. **Decision Boundary:** Colored contour regions demonstrating the learned boundary separating the two classes with test samples overlaid.
