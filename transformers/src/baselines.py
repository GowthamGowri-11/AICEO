"""
Statistical Baseline Models for Next-Token Prediction Benchmarking.

Baselines Implemented:
1. Uniform Random Guess Baseline: Probability 1 / |V|
2. Most Frequent Target Token Baseline: Always predicts the most common training target token
3. Unigram Statistical Baseline: Predicts target distribution matching training unigram frequencies
4. Bigram Markov Model: Maximum Likelihood Estimation of P(token_t | token_{t-1})
   with unigram backoff for unseen prefix tokens
"""

import math
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import DataLoader

PAD_ID = 0


class StatisticalBaselines:
    """
    Evaluates statistical n-gram and frequency baselines strictly fitted on training data.
    """

    def __init__(self, vocab_size: int, pad_id: int = PAD_ID):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.most_frequent_id: int = pad_id
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Dict[int, Counter] = defaultdict(Counter)
        self.total_train_targets: int = 0

    def fit(self, train_loader: DataLoader):
        """Fit unigram and bigram statistics strictly from the training DataLoader."""
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.total_train_targets = 0

        for inputs, targets in train_loader:
            for in_seq, tgt_seq in zip(inputs, targets):
                for inp, tgt in zip(in_seq.tolist(), tgt_seq.tolist()):
                    if tgt != self.pad_id:
                        self.unigram_counts[tgt] += 1
                        self.total_train_targets += 1
                        if inp != self.pad_id:
                            self.bigram_counts[inp][tgt] += 1

        if self.unigram_counts:
            self.most_frequent_id = self.unigram_counts.most_common(1)[0][0]

    def evaluate_on_loader(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate all baselines on the exact same held-out test targets.
        
        Returns:
            Dictionary containing accuracy, correct counts, and evaluated token counts.
        """
        total_eval_tokens = 0
        mf_correct = 0
        bigram_correct = 0

        for inputs, targets in test_loader:
            for in_seq, tgt_seq in zip(inputs, targets):
                for inp, tgt in zip(in_seq.tolist(), tgt_seq.tolist()):
                    if tgt != self.pad_id:
                        total_eval_tokens += 1

                        # 1. Most Frequent Token prediction
                        if self.most_frequent_id == tgt:
                            mf_correct += 1

                        # 2. Bigram Markov prediction
                        if inp in self.bigram_counts and len(self.bigram_counts[inp]) > 0:
                            bigram_pred = self.bigram_counts[inp].most_common(1)[0][0]
                        else:
                            bigram_pred = self.most_frequent_id  # Unigram backoff
                        if bigram_pred == tgt:
                            bigram_correct += 1

        # Uniform random accuracy
        uniform_acc = (1.0 / self.vocab_size) * 100 if self.vocab_size > 0 else 0.0
        mf_acc = (mf_correct / total_eval_tokens) * 100 if total_eval_tokens > 0 else 0.0
        bigram_acc = (bigram_correct / total_eval_tokens) * 100 if total_eval_tokens > 0 else 0.0

        return {
            "total_evaluated_tokens": total_eval_tokens,
            "vocabulary_size": self.vocab_size,
            "uniform_random_accuracy_pct": round(uniform_acc, 4),
            "most_frequent_token_id": self.most_frequent_id,
            "most_frequent_accuracy_pct": round(mf_acc, 2),
            "most_frequent_correct_tokens": mf_correct,
            "bigram_markov_accuracy_pct": round(bigram_acc, 2),
            "bigram_markov_correct_tokens": bigram_correct
        }
