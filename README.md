# Deep Learning Models & Architectures: ANN, CNN, and Transformers

A comprehensive, self-contained academic deep learning repository featuring three foundational neural network architectures built from first principles with full pipelines, evaluations, visualizations, and documentation.

---

## Repository Overview

```text
ANN-PROJECT/
├── ANN/             # Artificial Neural Network for Binary Classification (make_moons)
├── CNN/             # 3-Layer Convolutional Neural Network for Dog-Cat Image Classification
├── transformers/    # Causal Transformer Language Model with Scratch Self-Attention
├── .gitignore       # Repository exclusion rules
└── README.md        # Top-level repository overview
```

---

## 1. Project Modules

### 🔹 Module 1: Artificial Neural Network (ANN)
- **Directory:** [`ANN/`](ANN/)
- **Objective:** Learn non-linear decision boundaries on non-linearly separable 2D coordinates.
- **Dataset:** Scikit-learn `make_moons` dataset.
- **Architecture:** Feedforward Multi-Layer Perceptron (Input 2D $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(16, ReLU) $\rightarrow$ Dense(1, Sigmoid)).
- **Key Techniques:** Non-linear feature representation, binary cross-entropy loss, validation early stopping.
- **Deliverables:** Training script, decision boundary plots, and metrics.

### 🔹 Module 2: Convolutional Neural Network (CNN)
- **Directory:** [`CNN/`](CNN/)
- **Objective:** Binary classification of real Dog and Cat photographs.
- **Dataset:** Real `dogs_vs_cats_subset` photographs (800 train, 200 val, 200 test holdout, zero synthetic images).
- **Architecture:** Strictly 3-convolutional-layer CNN (Conv2D(32) $\rightarrow$ Conv2D(64) $\rightarrow$ Conv2D(128) $\rightarrow$ Flatten $\rightarrow$ Dense(128, L2=1e-4) $\rightarrow$ Dropout(0.4) $\rightarrow$ Sigmoid).
- **Key Results:**
  - **Test Accuracy:** **76.50%** on holdout unseen test set (reduced errors from 72 down to 47, a 34.7% error reduction over baseline).
  - **Cat Recall:** **84.00%** (up from 49.00%).
  - **Generalization Gap:** Reduced from 22.37% down to 3.25%, eliminating overfitting.
- **Inference Tool:** Standalone single-image predictor [`predict.py`](CNN/predict.py) with visual probability breakdown.

### 🔹 Module 3: Transformer Language Model with Custom Self-Attention
- **Directory:** [`transformers/`](transformers/)
- **Objective:** Next-token prediction and autoregressive text generation using a Transformer built from scratch.
- **Key Innovation:** Custom Scaled Dot-Product Attention ($Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$) implemented using pure PyTorch tensor operations.
- **Architecture:** Pre-LN Causal Transformer Decoder with $N=2$ blocks, $h=4$ attention heads, learnable positional embeddings, and weight tying.
- **Dataset:** Multi-domain educational English text corpus (science, technology, nature, education, geometry).
- **Key Results:**
  - **Test Accuracy:** **20.21%** on held-out test tokens (179× better than random chance baseline of 0.11% on 886-word vocabulary).
  - **Causal Invariance:** Verified by unit tests; future tokens cannot influence earlier representations.
  - **Explainability:** Real 2D causal attention weight heatmaps extracted from trained attention heads.
- **User Interface:** Streamlit web application [`app.py`](transformers/app.py) for interactive prompt testing and live attention visualization.

---

## 2. Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### Clone the Repository
```bash
git clone https://github.com/GowthamGowri-11/AICEO.git
cd AICEO
```

### Running Each Project
```bash
# 1. Run ANN Binary Classification
cd ANN
python main.py
cd ..

# 2. Run CNN Dog-Cat Classification
cd CNN
python main.py
python predict.py test_images/test_cat.jpg
cd ..

# 3. Run Transformer Next-Token Prediction
cd transformers
python main.py --mode all
streamlit run app.py
cd ..
```

---

## 3. License
All code and educational materials in this repository are open for academic and educational use.
