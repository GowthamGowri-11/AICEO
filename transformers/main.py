"""
Main Pipeline Entry Point for Transformer Next-Token Prediction Model.

Usage (from within the transformers/ folder):
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode predict --prompt "The sun rises in the"
    python main.py --mode visualize
    python main.py --mode test
    python main.py --mode all
"""

import os
import sys
import argparse
import json
import torch

# Ensure current directory is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from src.tokenizer import SimpleTokenizer
    from src.dataset import load_and_split_corpus, get_dataloaders
    from models.transformer_model import TransformerLM
    from src.train import train_model
    from src.evaluate import evaluate_test_set, load_model_from_checkpoint
    from src.predict import predict_next_token_distribution, generate_text
    from src.utils import set_seed, plot_loss_curves, plot_attention_map
    from tests.test_model import run_all_tests
except (ImportError, ModuleNotFoundError):
    from transformers.src.tokenizer import SimpleTokenizer
    from transformers.src.dataset import load_and_split_corpus, get_dataloaders
    from transformers.models.transformer_model import TransformerLM
    from transformers.src.train import train_model
    from transformers.src.evaluate import evaluate_test_set, load_model_from_checkpoint
    from transformers.src.predict import predict_next_token_distribution, generate_text
    from transformers.src.utils import set_seed, plot_loss_curves, plot_attention_map
    from transformers.tests.test_model import run_all_tests


# Default Configuration Constants
DEFAULT_CORPUS = os.path.join(CURRENT_DIR, "data", "raw", "corpus.txt")
DEFAULT_VOCAB = os.path.join(CURRENT_DIR, "data", "processed", "vocab.json")
DEFAULT_CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "models", "checkpoints")
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "best_transformer.pt")
DEFAULT_HISTORY = os.path.join(CURRENT_DIR, "results", "metrics", "history.json")
DEFAULT_METRICS = os.path.join(CURRENT_DIR, "results", "metrics", "test_metrics.json")
DEFAULT_LOSS_PLOT = os.path.join(CURRENT_DIR, "results", "plots", "loss_curves.png")
DEFAULT_ATTN_PLOT = os.path.join(CURRENT_DIR, "results", "plots", "attention_map.png")
DEFAULT_PREDICTIONS_TXT = os.path.join(CURRENT_DIR, "results", "sample_predictions.txt")

# Model Architecture Hyperparameters
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 32
D_FF = 256
DROPOUT = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 0.002
EPOCHS = 50
PATIENCE = 15


def run_training():
    """Execute end-to-end training pipeline."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/4] Preparing dataset from: {DEFAULT_CORPUS}")
    train_lines, val_lines, test_lines = load_and_split_corpus(DEFAULT_CORPUS, seed=42)

    print(f"  Training sentences:   {len(train_lines)}")
    print(f"  Validation sentences: {len(val_lines)}")
    print(f"  Test sentences:       {len(test_lines)}")

    print(f"\n[2/4] Fitting word-level tokenizer strictly on training split...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_lines)
    tokenizer.save(DEFAULT_VOCAB)
    print(f"  Vocabulary size: {tokenizer.vocab_size} tokens")
    print(f"  Vocabulary saved to: {DEFAULT_VOCAB}")

    print(f"\n[3/4] Creating causal next-token sequence dataloaders...")
    loaders = get_dataloaders(
        train_lines=train_lines,
        val_lines=val_lines,
        test_lines=test_lines,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN
    )
    print(f"  Training batches:   {len(loaders['train'])} ({len(loaders['train_ds'])} sequence pairs)")
    print(f"  Validation batches: {len(loaders['val'])} ({len(loaders['val_ds'])} sequence pairs)")
    print(f"  Test batches:       {len(loaders['test'])} ({len(loaders['test_ds'])} sequence pairs)")

    print(f"\n[4/4] Initializing Transformer language model...")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        d_ff=D_FF,
        dropout=DROPOUT,
        tie_weights=True
    )

    param_info = model.count_parameters()
    print(f"  Architecture: {NUM_LAYERS} Transformer Blocks, {NUM_HEADS} Attention Heads")
    print(f"  Embedding Dim: {D_MODEL}, Feed-Forward Dim: {D_FF}")
    print(f"  Total Parameters: {param_info['total']:,} (Trainable: {param_info['trainable']:,})")

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        tokenizer=tokenizer,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        history_path=DEFAULT_HISTORY,
        patience=PATIENCE
    )

    # Automatically generate loss plot after training
    plot_loss_curves(history, save_path=DEFAULT_LOSS_PLOT)
    return history


def run_evaluation():
    """Evaluate saved checkpoint on held-out test partition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from checkpoint: {DEFAULT_CHECKPOINT}")
    model, tokenizer, checkpoint = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device)

    train_lines, val_lines, test_lines = load_and_split_corpus(DEFAULT_CORPUS, seed=42)
    loaders = get_dataloaders(
        train_lines=train_lines,
        val_lines=val_lines,
        test_lines=test_lines,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN
    )

    print(f"Evaluating on {len(loaders['test_ds'])} held-out test sequence pairs...")
    metrics = evaluate_test_set(model, loaders["test"], tokenizer, device)

    metrics["best_epoch"] = checkpoint.get("epoch", "N/A")
    metrics["best_val_loss"] = checkpoint.get("val_loss", "N/A")
    metrics["dataset_splits"] = {
        "train_sentences": len(train_lines),
        "val_sentences": len(val_lines),
        "test_sentences": len(test_lines)
    }

    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_METRICS)), exist_ok=True)
    with open(DEFAULT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("           TRANSFORMER MODEL TEST EVALUATION")
    print("=" * 60)
    print(f"Test Loss:                   {metrics['test_loss']:.4f}")
    print(f"Test Perplexity:             {metrics['test_perplexity']:.4f}")
    print(f"Token-Level Accuracy:        {metrics['test_token_accuracy_pct']:.2f}%")
    print(f"Total Evaluated Tokens:      {metrics['total_evaluated_tokens']:,}")
    print(f"Correctly Predicted Tokens:  {metrics['total_correct_tokens']:,}")
    print(f"Vocabulary Size:             {metrics['vocabulary_size']}")
    print(f"Total Model Parameters:      {metrics['model_parameters']:,}")
    print(f"Best Validation Epoch:       {metrics['best_epoch']}")
    print("=" * 60)
    print(f"Detailed metrics saved to: {DEFAULT_METRICS}")
    return metrics


def run_prediction(prompt: str = "The sun rises in the", max_new_tokens: int = 8, temperature: float = 0.8):
    """Perform next-token prediction and text generation on an input prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device)

    top_candidates, _, input_tokens = predict_next_token_distribution(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device
    )

    print("\n" + "=" * 65)
    print("          TRANSFORMER NEXT-TOKEN INFERENCE REPORT")
    print("=" * 65)
    print(f"Input Prompt:        '{prompt}'")
    print(f"Normalized Tokens:   {input_tokens}")
    print("-" * 65)
    print("TOP-5 NEXT-TOKEN PREDICTIONS:")
    for rank, (token, prob) in enumerate(top_candidates, 1):
        bar_len = int(round(prob * 30))
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {rank}. {token:<16} : {prob * 100:6.2f}%  [{bar}]")
    print("-" * 65)

    best_next_token = top_candidates[0][0]
    print(f"Selected Next Token: >>> '{best_next_token}' <<<")

    # Greedy Autoregressive Generation
    greedy_text, _ = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        sample=False,
        device=device
    )
    print(f"Greedy Completion:   '{greedy_text}'")

    # Sampled Autoregressive Generation
    sampled_text, _ = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        sample=True,
        device=device
    )
    print(f"Sampled (T={temperature}):    '{sampled_text}'")
    print("=" * 65)


def run_sample_predictions():
    """Run a diverse batch of academic prompts and record to sample_predictions.txt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device)

    test_prompts = [
        "The sun rises in the",
        "Machine learning is a",
        "Water freezes into solid",
        "Trees produce oxygen through",
        "Students attend schools and",
        "A triangle is a geometric",
        "Artificial intelligence aims to",
        "Light travels faster than"
    ]

    output_lines = [
        "=" * 70,
        "   TRANSFORMER NEXT-TOKEN PREDICTION SAMPLE REPORT",
        "=" * 70,
        ""
    ]

    for prompt in test_prompts:
        top_candidates, _, _ = predict_next_token_distribution(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device
        )
        completion, _ = generate_text(
            model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=6, sample=False, device=device
        )

        output_lines.append(f"Prompt:     '{prompt}'")
        output_lines.append(f"Top Next:   '{top_candidates[0][0]}' ({top_candidates[0][1]*100:.2f}%)")
        output_lines.append(f"Top 3:      1. {top_candidates[0][0]} ({top_candidates[0][1]*100:.1f}%), "
                            f"2. {top_candidates[1][0]} ({top_candidates[1][1]*100:.1f}%), "
                            f"3. {top_candidates[2][0]} ({top_candidates[2][1]*100:.1f}%)")
        output_lines.append(f"Completion: '{completion}'")
        output_lines.append("-" * 70)

    report_str = "\n".join(output_lines)
    os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_PREDICTIONS_TXT)), exist_ok=True)
    with open(DEFAULT_PREDICTIONS_TXT, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(report_str)
    print(f"\nSample predictions saved to: {DEFAULT_PREDICTIONS_TXT}")


def run_visualization():
    """Generate loss curve and attention map plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Loss curves from history
    if os.path.exists(DEFAULT_HISTORY):
        with open(DEFAULT_HISTORY, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_loss_curves(history, save_path=DEFAULT_LOSS_PLOT)
    else:
        print(f"[Warning] History file not found at {DEFAULT_HISTORY}. Train the model first.")

    # 2. Extract real attention weights from trained model
    if os.path.exists(DEFAULT_CHECKPOINT):
        model, tokenizer, _ = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device)
        sample_sentence = "the sun rises in the east"
        tokens = tokenizer.tokenize(sample_sentence)
        ids = tokenizer.encode(sample_sentence, add_bos=True, add_eos=False)
        display_tokens = [tokenizer.bos_token] + tokens

        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, attentions = model(input_tensor, return_attention=True)

        if attentions and len(attentions) > 0:
            # First layer, first head attention weights: shape (1, num_heads, seq_len, seq_len)
            attn_matrix = attentions[0][0, 0].cpu().numpy()
            plot_attention_map(
                attention_matrix=attn_matrix,
                tokens=display_tokens,
                layer_idx=0,
                head_idx=0,
                save_path=DEFAULT_ATTN_PLOT
            )
    else:
        print(f"[Warning] Checkpoint not found at {DEFAULT_CHECKPOINT}.")


def main():
    parser = argparse.ArgumentParser(
        description="Transformer Next-Token Prediction with Scratch Self-Attention."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict", "visualize", "test", "all"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The sun rises in the",
        help="Input text prompt for prediction mode"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=8,
        help="Number of tokens to generate in predict mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for text generation"
    )
    args = parser.parse_args()

    if args.mode == "test":
        run_all_tests()
    elif args.mode == "train":
        run_training()
    elif args.mode == "evaluate":
        run_evaluation()
    elif args.mode == "predict":
        run_prediction(args.prompt, max_new_tokens=args.tokens, temperature=args.temperature)
    elif args.mode == "visualize":
        run_visualization()
    elif args.mode == "all":
        print("\n=== [STEP 1/5] RUNNING TEST SUITE ===")
        run_all_tests()
        print("\n=== [STEP 2/5] TRAINING MODEL ===")
        run_training()
        print("\n=== [STEP 3/5] EVALUATING TEST SET ===")
        run_evaluation()
        print("\n=== [STEP 4/5] GENERATING VISUALIZATIONS ===")
        run_visualization()
        print("\n=== [STEP 5/5] GENERATING SAMPLE PREDICTIONS ===")
        run_sample_predictions()
        print("\n=== FULL PIPELINE COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
