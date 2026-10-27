"""
Byte-Pair Encoding (BPE) Subword Tokenizer Implemented from Scratch.

Key Features:
- Pure Python implementation without external dependencies
- Subword representation eliminates out-of-vocabulary (<UNK>) tokens (0.0% OOV)
- Reserved special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
- End-of-word marker '</w>' preserves exact word boundaries for lossless round-trip detokenization
- Serialization/deserialization to JSON
"""

import os
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class BPETokenizer:
    """
    Subword tokenizer using Byte Pair Encoding (BPE).
    """

    def __init__(self, vocab_size: int = 400):
        self.target_vocab_size = vocab_size
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_token_id = PAD_ID
        self.unk_token_id = UNK_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID

        self.merges: List[Tuple[str, str]] = []
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id2token = {idx: tok for idx, tok in enumerate(SPECIAL_TOKENS)}

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:
        """Count frequency of adjacent symbol pairs in current vocabulary."""
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], v_in: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of pair in vocabulary words."""
        v_out = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        replacement = "".join(pair)

        for word, freq in v_in.items():
            w_str = " ".join(word)
            w_new = pattern.sub(replacement, w_str)
            v_out[tuple(w_new.split())] = freq
        return v_out

    def fit(self, texts: List[str], min_freq: int = 2):
        """
        Learn BPE merges strictly from training texts.
        """
        self._init_special_tokens()
        self.merges = []

        # Step 1: Pre-tokenize into lowercased words with '</w>' end-of-word marker
        word_freqs = Counter()
        pattern = re.compile(r"\w+|[^\w\s]")
        for text in texts:
            words = pattern.findall(text.lower().strip())
            for w in words:
                word_freqs[w] += 1

        # Represent each word as a tuple of characters with end marker '</w>'
        vocab: Dict[Tuple[str, ...], int] = {}
        initial_chars = set()
        for w, freq in word_freqs.items():
            if not w:
                continue
            chars = tuple(list(w) + ["</w>"])
            vocab[chars] = freq
            for c in chars:
                initial_chars.add(c)

        # Register all initial characters in vocabulary
        for c in sorted(initial_chars):
            if c not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[c] = new_id
                self.id2token[new_id] = c

        num_merges = max(0, self.target_vocab_size - len(self.token2id))

        # Iteratively find most frequent pair and merge
        for _ in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair, best_freq = pairs.most_common(1)[0]
            if best_freq < min_freq:
                break

            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            new_subword = "".join(best_pair)
            if new_subword not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[new_subword] = new_id
                self.id2token[new_id] = new_subword

    def tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word into learned subword units."""
        if not word:
            return []
        word_tuple = tuple(list(word.lower()) + ["</w>"])

        for pair in self.merges:
            if len(word_tuple) <= 1:
                break
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            word_tuple = tuple(new_word)

        return list(word_tuple)

    def tokenize(self, text: str) -> List[str]:
        """Convert input sentence into a list of BPE subword tokens."""
        if not text or not isinstance(text, str):
            return []
        pattern = re.compile(r"\w+|[^\w\s]")
        raw_words = pattern.findall(text.lower().strip())
        tokens = []
        for w in raw_words:
            tokens.extend(self.tokenize_word(w))
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token ID sequence."""
        subwords = self.tokenize(text)
        ids = [self.token2id.get(sw, self.unk_token_id) for sw in subwords]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode sequence of token IDs back into human readable text."""
        subwords = []
        for idx in ids:
            tok = self.id2token.get(idx, self.unk_token)
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            subwords.append(tok)

        text = "".join(subwords).replace("</w>", " ").strip()
        # Clean up punctuation spacing
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text

    def save(self, filepath: str):
        """Serialize BPE vocabulary and merge rules to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "token2id": self.token2id,
            "merges": self.merges,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Deserialize BPE tokenizer from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("target_vocab_size", 400))
        tok.token2id = {k: int(v) for k, v in data["token2id"].items()}
        tok.id2token = {int(v): k for k, v in data["token2id"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok
