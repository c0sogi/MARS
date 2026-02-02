import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config


class Tokenizer:
    """
    A simple tokenizer that maps words to indices and handles embedding matrices.
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_size = 0
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNKNOWN_TOKEN

    def fit_on_texts(
        self, texts, max_vocab_size=Config.MAX_VOCAB_SIZE, min_freq=Config.MIN_FREQ
    ):
        """
        Builds vocabulary from a list of strings.
        """
        token_counts = Counter()
        for text in texts:
            tokens = text.split()
            token_counts.update(tokens)

        # Start with special tokens
        self.idx2word = [self.pad_token, self.unk_token]
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}

        # Add most common words
        # We subtract 2 from max_vocab_size to account for PAD and UNK
        most_common = token_counts.most_common(max_vocab_size - 2)

        for word, count in most_common:
            if count >= min_freq:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

        self.vocab_size = len(self.idx2word)
        print(f"Tokenizer fitted. Vocab size: {self.vocab_size}")

    def text_to_sequence(self, text, max_len=None):
        """
        Converts a single string to a list of indices.
        """
        tokens = text.split()
        sequence = [
            self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens
        ]

        if max_len:
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            else:
                sequence = sequence + [self.word2idx[self.pad_token]] * (
                    max_len - len(sequence)
                )

        return sequence

    def texts_to_sequences(self, texts, max_len=None):
        """
        Converts a list of strings to a list of list of indices.
        """
        return [self.text_to_sequence(text, max_len) for text in texts]

    def save(self, path):
        """
        Saves the vocabulary (idx2word list) to a .npy file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.idx2word))
        print(f"Tokenizer vocabulary saved to {path}")

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        self.idx2word = np.load(path, allow_pickle=True).tolist()
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.idx2word)
        print(f"Tokenizer vocabulary loaded from {path}. Size: {self.vocab_size}")

    def create_embedding_matrix(
        self, embedding_dim=Config.EMBEDDING_DIM, glove_path=None
    ):
        """
        Creates an embedding matrix. Loads GloVe if path provided, else random initialization.
        """
        if self.vocab_size == 0:
            raise ValueError(
                "Tokenizer must be fitted or loaded before creating embedding matrix."
            )

        # Initialize with random values (normal distribution)
        embedding_matrix = np.random.normal(
            scale=0.6, size=(self.vocab_size, embedding_dim)
        )

        # Zero out the padding token
        if self.pad_token in self.word2idx:
            embedding_matrix[self.word2idx[self.pad_token]] = np.zeros(embedding_dim)

        if glove_path and os.path.exists(glove_path):
            print(f"Loading GloVe embeddings from {glove_path}...")
            hits = 0
            with open(glove_path, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in self.word2idx:
                        try:
                            vector = np.asarray(values[1:], dtype="float32")
                            if len(vector) == embedding_dim:
                                embedding_matrix[self.word2idx[word]] = vector
                                hits += 1
                        except ValueError:
                            continue
            print(
                f"Loaded {hits} vectors from GloVe. Coverage: {hits/self.vocab_size:.2%}"
            )
        else:
            print(
                "No GloVe path provided or file not found. Using random initialization."
            )

        return embedding_matrix.astype(np.float32)


def segment_sentences(document_text):
    """
    Splits document text into sentences based on punctuation and HTML tags.
    Returns a list of dicts: {'text': str, 'start_token_idx': int, 'end_token_idx': int}
    """
    tokens = document_text.split()
    sentences = []
    current_sent_tokens = []
    start_idx = 0

    # Regex to identify HTML tags (e.g., <P>, </H1>)
    tag_pattern = re.compile(r"^<[^>]+>$")

    # Common sentence terminators
    terminators = {".", "?", "!"}

    for i, token in enumerate(tokens):
        current_sent_tokens.append(token)

        # Determine if this token ends a sentence
        is_tag = bool(tag_pattern.match(token))
        ends_with_punct = token[-1] in terminators

        # Split if it's a tag or ends with punctuation
        # Note: We treat tags as separate structural "sentences" or boundaries
        if is_tag or ends_with_punct:
            end_idx = i + 1
            sentences.append(
                {
                    "text": " ".join(current_sent_tokens),
                    # 'tokens': current_sent_tokens, # Optional: keep raw tokens if needed
                    "start_token_idx": start_idx,
                    "end_token_idx": end_idx,
                }
            )
            current_sent_tokens = []
            start_idx = end_idx

    # Flush any remaining tokens as the last sentence
    if current_sent_tokens:
        end_idx = len(tokens)
        sentences.append(
            {
                "text": " ".join(current_sent_tokens),
                "start_token_idx": start_idx,
                "end_token_idx": end_idx,
            }
        )

    return sentences


def build_or_load_tokenizer(
    load_cached_data=True,
    train_data_path=Config.TRAIN_DATA_PATH,
    sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Builds a tokenizer from training data or loads it from cache.
    """
    vocab_path = Config.VOCAB_PATH
    tokenizer = Tokenizer()

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(vocab_path):
        try:
            tokenizer.load(vocab_path)
            return tokenizer
        except Exception as e:
            print(f"Failed to load cached tokenizer: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Building tokenizer from training data...")

    # We need to read the documents to build vocab.
    # To save memory, we can stream the file or read chunks.
    texts = []

    # Read training data
    # If sample_size is set, we only read that many lines
    count = 0
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if sample_size and count >= sample_size:
                break

            entry = json.loads(line)
            # Add document text
            texts.append(entry.get("document_text", ""))
            # Add question text
            texts.append(entry.get("question_text", ""))
            count += 1

            if count % 10000 == 0:
                print(f"Processed {count} examples for vocabulary...")

    tokenizer.fit_on_texts(texts)

    # 3. Save to cache
    tokenizer.save(vocab_path)

    return tokenizer


def build_or_load_embedding_matrix(tokenizer, load_cached_data=True, glove_path=None):
    """
    Creates embedding matrix or loads from cache.
    """
    matrix_path = Config.EMBEDDING_MATRIX_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(matrix_path):
        try:
            matrix = np.load(matrix_path)
            if matrix.shape == (tokenizer.vocab_size, Config.EMBEDDING_DIM):
                print(
                    f"Embedding matrix loaded from {matrix_path}. Shape: {matrix.shape}"
                )
                return matrix
            else:
                print(
                    f"Cached embedding matrix shape {matrix.shape} mismatch with vocab {tokenizer.vocab_size}. Rebuilding..."
                )
        except Exception as e:
            print(f"Failed to load cached embedding matrix: {e}. Rebuilding...")

    # 2. Build from scratch
    print("Creating embedding matrix...")
    matrix = tokenizer.create_embedding_matrix(
        embedding_dim=Config.EMBEDDING_DIM, glove_path=glove_path
    )

    # 3. Save to cache
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)
    print(f"Embedding matrix saved to {matrix_path}")

    return matrix
