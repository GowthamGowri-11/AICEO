# Development of a Transformer-Based Next-Token Prediction Model Using Basic Self-Attention

A comprehensive, self-contained academic deep learning repository featuring a causal Transformer language model implemented from first principles in pure PyTorch. The central multi-head scaled dot-product attention mechanism, causal lower-triangular masking, learnable positional embeddings, and Transformer blocks are built from tensor operations without relying on prebuilt Transformer libraries or pretrained models.

---

## Repository Structure

```text
ANN-PROJECT/
├── transformers/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── corpus.txt              # Original multi-domain training corpus
│   │   │   └── corpus_expanded.txt     # Expanded educational training corpus (903 sentences)
│   │   └── processed/
│   │       ├── vocab.json              # Serialized baseline vocabulary (886 tokens)
│   │       └── vocab_improved.json     # Serialized expanded vocabulary (3,189 tokens)
│   ├── models/
│   │   ├── transformer_model.py        # Pre-LN Causal TransformerLM neural network architecture
│   │   └── checkpoints/
│   │       ├── best_transformer.pt     # Preserved original baseline production checkpoint
│   │       └── best_transformer_improved.pt # Improved production checkpoint with padding masks
│   ├── src/
│   │   ├── __init__.py
│   │   ├── tokenizer.py                # Normalized word-level tokenizer with punctuation support
│   │   ├── bpe_tokenizer.py            # Custom Byte-Pair Encoding (BPE) subword tokenizer (0.0% OOV)
│   │   ├── dataset.py                  # Causal sequence alignment and PyTorch DataLoaders
│   │   ├── positional_encoding.py      # Learnable and sinusoidal positional embeddings
│   │   ├── self_attention.py           # Core scaled dot-product causal self-attention
│   │   ├── transformer_block.py        # Pre-LN Transformer decoder blocks (Attn + FFN)
│   │   ├── train.py                    # Training loop with validation and early stopping
│   │   ├── evaluate.py                 # Quantitative test set evaluation and perplexity calculation
│   │   ├── baselines.py                # Statistical baselines (Uniform, Most Frequent, Bigram Markov)
│   │   ├── predict.py                  # Next-token prediction, greedy decoding, and nucleus sampling
│   │   └── utils.py                    # Loss curve plotting and attention heatmap generation
│   ├── results/
│   │   ├── plots/
│   │   │   ├── loss_curves.png         # Original baseline training loss curves
│   │   │   ├── improved_loss_curves.png # Improved model loss curves
│   │   │   └── attention_map.png       # Extracted causal attention weight heatmap
│   │   ├── metrics/
│   │   │   ├── test_metrics.json       # Original baseline evaluation metrics
│   │   │   ├── improved_test_metrics.json # Improved model evaluation metrics
│   │   │   ├── baseline_comparison.json   # Statistical baselines benchmark comparison
│   │   │   └── experiment_summary.json    # Ablation and multi-experiment summary
│   │   ├── sample_predictions.txt      # Original baseline qualitative prediction report
│   │   └── improved_sample_predictions.txt # Improved model qualitative prediction report
│   ├── tests/
│   │   └── test_model.py               # Unit test suite (8/8 tests: tokenizers, causal masks, invariance)
│   ├── app.py                          # Streamlit web application with live checkpoint switching
│   ├── main.py                         # Unified command-line pipeline runner
│   └── requirements.txt                # Python dependencies
├── .gitignore                          # Repository exclusion rules
└── README.md                           # Project documentation
```

---

## 1. Key Architectural Specifications

- **Architecture:** Pre-LayerNorm (Pre-LN) Causal Transformer Decoder
- **Blocks ($N$):** 2 Transformer blocks
- **Attention Heads ($h$):** 4 parallel attention heads ($d_k = d_v = 16$, $d_{\text{model}} = 64$)
- **Feed-Forward Dimension ($d_{\text{ff}}$):** 256 (GELU activation)
- **Positional Encoding:** Learnable positional embeddings ($T_{\max} = 32$)
- **Weight Tying:** Shared parameter matrix between token embedding and LM head
- **Key-Padding Masking:** Automatic propagation through all attention heads
- **Causal Invariance:** Mathematically verified (future token alterations cause $< 10^{-6}$ change in earlier token representations)

---

## 2. Quantitative Performance & Statistical Baselines

Evaluated on held-out test splits with non-padding target tokens:

| Metric | Original Baseline Model | Improved Model (`best_transformer_improved.pt`) |
| :--- | :---: | :---: |
| **Training Sentences** | 157 | **722** |
| **Evaluated Test Tokens** | 287 | **1,560** |
| **Vocabulary Size** | 886 tokens | **3,189 tokens** |
| **Test Out-of-Vocabulary (OOV) Rate** | **28.20%** | **16.20%** |
| **Subword BPE Test OOV Rate** | *N/A* | **0.00%** (via `src/bpe_tokenizer.py`) |
| **Test Token Accuracy (Top-1)** | 20.21% (58 / 287) | **16.92%** (264 / 1,560) |
| **Top-5 Test Accuracy** | *Not Measured* | **26.79%** (418 / 1,560) |
| **Bigram Markov Baseline** | 19.86% (57 / 287) | **17.18%** (268 / 1,560) |
| **Most Frequent Token Baseline** | 7.32% (21 / 287) | **5.83%** (91 / 1,560) |
| **Uniform Random Guess Baseline** | 0.11% ($1/886$) | **0.0314%** ($1/3189$) |
| **Parameters** | 158,336 | **305,728** |

---

## 3. Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- PyTorch 2.0+

### Installation & Execution (Windows PowerShell)

```powershell
# 1. Enter the project directory
cd transformers

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the automated unit test suite (8/8 tests)
python -m pytest tests/test_model.py -v

# 4. Run baseline test evaluation
python main.py --mode evaluate

# 5. Interactive prompt inference
python main.py --mode predict --prompt "The sun rises in the"

# 6. Launch the Streamlit interactive web application
streamlit run app.py
```

The Streamlit web UI allows you to:
- Interactively switch between `best_transformer_improved.pt` and `best_transformer.pt`.
- Type custom prompts and inspect Top-5 probability distributions.
- Adjust generation parameters (Greedy, Sampling, Top-K, Nucleus Top-P).
- Visualize extracted 2D causal self-attention heatmaps across different layers and heads.

---

## 4. License
All code and educational materials in this repository are open for academic and educational use.
