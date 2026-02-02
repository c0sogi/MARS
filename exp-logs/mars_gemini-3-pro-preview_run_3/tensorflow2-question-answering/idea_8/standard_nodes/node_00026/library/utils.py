import os
import re
import string
import numpy as np
from collections import Counter
from library.config import Config


def tokenize_text(text):
    """
    Splits text into tokens based on whitespace.

    Args:
        text (str): Input text.

    Returns:
        list: List of tokens.
    """
    if not text:
        return []
    return text.split()


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Standard normalization for SQuAD/NQ evaluation.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    Computes exact match score between prediction and ground truth.

    Args:
        prediction (str): The predicted answer text.
        ground_truth (str): The true answer text.

    Returns:
        float: 1.0 if match, 0.0 otherwise.
    """
    return (
        1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    )


def f1_score(prediction, ground_truth):
    """
    Computes F1 score based on token overlap.

    Args:
        prediction (str): The predicted answer text.
        ground_truth (str): The true answer text.

    Returns:
        float: The F1 score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def parse_html_candidates(tokens):
    """
    Segments document tokens into candidate paragraphs based on top-level HTML tags.
    Handles nested tags by only closing a candidate when the top-level tag closes.

    Args:
        tokens (list): List of document tokens.

    Returns:
        list: A list of tuples (start_token_idx, end_token_idx) representing candidates.
    """
    candidates = []
    stack = []
    start_index = -1

    # Set of tags that define a candidate block
    # Note: NQ simplified HTML uses these tags.
    block_tags = {
        "<P>",
        "<Table>",
        "<Ul>",
        "<Ol>",
        "<Dl>",
        "<H1>",
        "<H2>",
        "<H3>",
        "<H4>",
        "<H5>",
        "<H6>",
    }

    for i, token in enumerate(tokens):
        # Check if token is a start tag
        if token in block_tags:
            # If stack is empty, this is a top-level start
            if not stack:
                start_index = i
            stack.append(token)

        # Check if token is an end tag corresponding to a block tag
        elif token.startswith("</") and token.endswith(">"):
            # Extract tag name to match (e.g., </P> -> <P>)
            tag_name = token[2:-1]
            start_tag = f"<{tag_name}>"

            if start_tag in block_tags:
                # If this closes the most recent open tag
                if stack and stack[-1] == start_tag:
                    stack.pop()
                    # If stack is now empty, we closed the top-level tag
                    if not stack and start_index != -1:
                        candidates.append((start_index, i + 1))
                        start_index = -1

    # Handle case where a tag might not be closed at the very end (rare in NQ)
    if stack and start_index != -1:
        candidates.append((start_index, len(tokens)))

    return candidates


def load_glove_embeddings(
    vocab, embedding_dim, glove_file_path=None, load_cached_data=True
):
    """
    Loads embedding matrix. Uses caching to speed up subsequent runs.

    Args:
        vocab (dict): Dictionary mapping tokens to indices.
        embedding_dim (int): Dimension of embeddings (e.g., 100).
        glove_file_path (str, optional): Path to raw GloVe text file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        numpy.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    cache_path = Config.EMBEDDING_MATRIX_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute/Initialize embeddings
    print("Initializing embedding matrix...")
    vocab_size = len(vocab)

    # Initialize with random normal distribution
    embedding_matrix = np.random.normal(
        scale=0.6, size=(vocab_size, embedding_dim)
    ).astype(np.float32)

    # If a GloVe file is provided and exists, load it
    if glove_file_path and os.path.exists(glove_file_path):
        print(f"Parsing GloVe file: {glove_file_path}")
        hits = 0
        with open(glove_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                if word in vocab:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if vector.shape[0] == embedding_dim:
                        embedding_matrix[vocab[word]] = vector
                        hits += 1
        print(f"Loaded {hits} vectors from GloVe.")
    else:
        print("No GloVe file provided or found. Using random initialization.")

    # 3. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embedding_matrix)
        print(f"Saved embedding matrix to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save embedding cache: {e}")

    return embedding_matrix
