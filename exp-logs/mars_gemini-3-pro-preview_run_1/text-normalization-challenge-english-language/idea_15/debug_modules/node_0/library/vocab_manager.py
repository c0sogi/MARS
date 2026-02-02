import os
import json
import pandas as pd
from collections import Counter
import sentencepiece as spm
from typing import List, Dict, Tuple, Optional, Union, Iterable

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("vocab_manager")


class Vocab:
    """
    Generic Vocabulary class to map tokens to indices and vice versa.
    """

    def __init__(self, name: str, specials: List[str] = None):
        self.name = name
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.specials = specials if specials else []

        # Initialize with specials
        for token in self.specials:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """Adds a token to the vocabulary if it doesn't exist."""
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token
        return self.stoi[token]

    def add_tokens(self, tokens: Iterable[str]):
        """Adds a list of tokens to the vocabulary."""
        for token in tokens:
            self.add_token(token)

    def __len__(self) -> int:
        return len(self.stoi)

    def __getitem__(self, token: str) -> int:
        """
        Returns the index of the token.
        Returns index of <unk> if token not found and <unk> exists.
        """
        if token in self.stoi:
            return self.stoi[token]

        if "<unk>" in self.stoi:
            return self.stoi["<unk>"]

        # If no <unk>, raise error or return a default (here we raise error for safety)
        raise KeyError(
            f"Token '{token}' not found in vocabulary '{self.name}' and no <unk> token defined."
        )

    def lookup_token(self, idx: int) -> str:
        """Returns the token for a given index."""
        if idx in self.itos:
            return self.itos[idx]
        raise KeyError(f"Index {idx} not found in vocabulary '{self.name}'.")

    def save(self, filepath: str):
        """Saves the stoi dictionary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            # Save specials separately to ensure they are loaded first/correctly if needed,
            # but saving the full stoi is usually sufficient for reconstruction.
            data = {"name": self.name, "specials": self.specials, "stoi": self.stoi}
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved vocabulary '{self.name}' to {filepath} (Size: {len(self)})")

    def load(self, filepath: str):
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.name = data.get("name", self.name)
        self.specials = data.get("specials", [])
        self.stoi = data["stoi"]
        self.itos = {int(k): v for v, k in self.stoi.items()}  # Reconstruct itos
        logger.info(
            f"Loaded vocabulary '{self.name}' from {filepath} (Size: {len(self)})"
        )


def train_bpe_tokenizer(text_corpus: List[str], model_prefix: str, vocab_size: int):
    """
    Trains a SentencePiece BPE tokenizer.
    """
    # Create a temporary corpus file
    corpus_path = os.path.join(Config.WORKING_DIR, "bpe_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in text_corpus:
            f.write(line + "\n")

    logger.info(f"Training BPE tokenizer with vocab size {vocab_size}...")

    # Train SentencePiece
    # user_defined_symbols can be added if we want to preserve specific tokens
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,  # Ensure all chars are covered or fall back to byte fallback
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>",
    )
    logger.info(f"BPE tokenizer trained and saved to {model_prefix}.model")


def build_vocabs(
    load_cached_data: bool = True,
) -> Tuple[Vocab, Vocab, Vocab, spm.SentencePieceProcessor]:
    """
    Orchestrates the creation or loading of all vocabularies.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        word_vocab (Vocab): Vocabulary for input words.
        char_vocab (Vocab): Vocabulary for characters (input and output).
        class_vocab (Vocab): Vocabulary for target classes.
        bpe_tokenizer (spm.SentencePieceProcessor): Trained BPE tokenizer.
    """

    # Define file paths
    word_vocab_path = Config.WORD_VOCAB_FILE
    char_vocab_path = Config.CHAR_VOCAB_FILE
    class_vocab_path = Config.CLASS_VOCAB_FILE
    bpe_model_path = Config.BPE_MODEL_FILE

    # Check if all artifacts exist
    all_exist = (
        os.path.exists(word_vocab_path)
        and os.path.exists(char_vocab_path)
        and os.path.exists(class_vocab_path)
        and os.path.exists(bpe_model_path)
    )

    if load_cached_data and all_exist:
        logger.info("Loading vocabularies from cache...")

        # Load Word Vocab
        word_vocab = Vocab("word_vocab")
        word_vocab.load(word_vocab_path)

        # Load Char Vocab
        char_vocab = Vocab("char_vocab")
        char_vocab.load(char_vocab_path)

        # Load Class Vocab
        class_vocab = Vocab("class_vocab")
        class_vocab.load(class_vocab_path)

        # Load BPE
        bpe_tokenizer = spm.SentencePieceProcessor()
        bpe_tokenizer.load(bpe_model_path)

        return word_vocab, char_vocab, class_vocab, bpe_tokenizer

    # --- Build from Scratch ---
    logger.info("Building vocabularies from scratch...")

    # Ensure output directory exists
    os.makedirs(Config.VOCAB_DIR, exist_ok=True)

    # Load Training Data
    logger.info(f"Reading training data from {Config.TRAIN_FILE}...")
    df = pd.read_csv(Config.TRAIN_FILE, dtype=str, keep_default_na=False)

    # 1. Build Word Vocabulary
    # ------------------------
    logger.info("Building Word Vocabulary...")
    word_vocab = Vocab("word_vocab", specials=["<pad>", "<unk>"])

    # Count word frequencies
    tokens = df["before"].astype(str).tolist()
    counter = Counter(tokens)

    # Add most common words up to limit
    most_common = counter.most_common(Config.MAX_WORD_VOCAB_SIZE)
    words = [token for token, count in most_common]
    word_vocab.add_tokens(words)
    word_vocab.save(word_vocab_path)

    # 2. Build Character Vocabulary
    # -----------------------------
    logger.info("Building Character Vocabulary...")
    char_vocab = Vocab("char_vocab", specials=["<pad>", "<unk>", "<sos>", "<eos>"])

    # Collect unique characters from both 'before' and 'after'
    # Using set for uniqueness
    input_chars = set("".join(df["before"].astype(str).unique()))
    target_chars = set("".join(df["after"].astype(str).unique()))
    all_chars = sorted(list(input_chars.union(target_chars)))

    char_vocab.add_tokens(all_chars)
    char_vocab.save(char_vocab_path)

    # 3. Build Class Vocabulary
    # -------------------------
    logger.info("Building Class Vocabulary...")
    class_vocab = Vocab(
        "class_vocab", specials=[]
    )  # Classes usually don't need <pad>/<unk> if strictly defined

    unique_classes = sorted(df["class"].astype(str).unique().tolist())
    class_vocab.add_tokens(unique_classes)
    class_vocab.save(class_vocab_path)

    # 4. Train BPE Tokenizer
    # ----------------------
    logger.info("Training BPE Tokenizer...")
    # We use the unique 'before' tokens to train the BPE to avoid redundancy and speed up training,
    # or use the full corpus if frequency matters for BPE split logic.
    # Usually full corpus is better for BPE frequency stats.
    # However, given 7M rows, we might sample or just use unique tokens weighted by count if supported.
    # Standard SP training takes a file. We'll use the full 'before' column.

    # Note: SentencePiece trainer might be slow on 7M lines.
    # We can use a subset if DEBUG is on, but here we assume full build.
    if Config.DEBUG:
        train_texts = df["before"].astype(str).iloc[: Config.DEBUG_SIZE].tolist()
    else:
        train_texts = df["before"].astype(str).tolist()

    train_bpe_tokenizer(
        text_corpus=train_texts,
        model_prefix=Config.BPE_MODEL_PREFIX,
        vocab_size=Config.BPE_VOCAB_SIZE,
    )

    # Load the trained model
    bpe_tokenizer = spm.SentencePieceProcessor()
    bpe_tokenizer.load(bpe_model_path)

    logger.info("All vocabularies built and saved.")
    return word_vocab, char_vocab, class_vocab, bpe_tokenizer
