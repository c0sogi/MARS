import os
import random
import numpy as np
import torch
import logging
import sys
import re
import string
from collections import Counter
from library.config import TrainingConfig


# Define HTML tags globally to ensure consistency across modules
HTML_TAGS = {
    "<P>",
    "</P>",
    "<H1>",
    "</H1>",
    "<H2>",
    "</H2>",
    "<H3>",
    "</H3>",
    "<H4>",
    "</H4>",
    "<H5>",
    "</H5>",
    "<H6>",
    "</H6>",
    "<Ul>",
    "</Ul>",
    "<Ol>",
    "</Ol>",
    "<Dl>",
    "</Dl>",
    "<Table>",
    "</Table>",
    "<Tr>",
    "</Tr>",
    "<Td>",
    "</Td>",
    "<Th>",
    "</Th>",
    "<Li>",
    "</Li>",
    "<Dd>",
    "</Dd>",
    "<Dt>",
    "</Dt>",
    # Added tags to pass tests and handle common formatting
    "<B>",
    "</B>",
    "<I>",
    "</I>",
}


def set_seed(seed=TrainingConfig.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(name="idea_6", log_file=None):
    """
    Sets up a logger with the specified name and optional file output.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to a file to log to.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def parse_html(document_text):
    """
    Segments document text into candidate paragraphs based on HTML tags and cleans them.
    This function splits the tokenized text into chunks defined by block-level HTML tags
    and removes the tags from the text content.

    Args:
        document_text (str): The raw document text containing HTML tags (space-separated tokens).

    Returns:
        List[Dict]: A list of candidate dictionaries, each containing:
            - 'text': The cleaned text of the paragraph (str).
            - 'start_token': The starting token index in the original document (int).
            - 'end_token': The ending token index in the original document (int).
    """
    tokens = document_text.split()
    candidates = []

    # Block level tags that signal the start of a new semantic segment
    block_start_tags = {
        "<P>",
        "<H1>",
        "<H2>",
        "<H3>",
        "<H4>",
        "<H5>",
        "<H6>",
        "<Li>",
        "<Dd>",
        "<Dt>",
        "<Tr>",
    }

    current_tokens = []
    start_index = 0
    in_candidate = False

    for i, token in enumerate(tokens):
        is_tag = token in HTML_TAGS

        # Heuristic: Start a new candidate on block start tags
        if token in block_start_tags:
            # If we were already building a candidate, save it
            if current_tokens:
                clean_text = " ".join(current_tokens)
                if clean_text.strip():
                    candidates.append(
                        {"text": clean_text, "start_token": start_index, "end_token": i}
                    )
                current_tokens = []

            start_index = i
            in_candidate = True
            continue  # Skip adding the tag itself to the text content

        if not is_tag:
            if not in_candidate:
                # If we encounter text but aren't in a candidate (e.g. start of doc), start one
                start_index = i
                in_candidate = True
            current_tokens.append(token)
        else:
            # It is a tag, skip adding it to current_tokens but don't break the segment
            pass

    # Add the last candidate if exists
    if current_tokens:
        clean_text = " ".join(current_tokens)
        if clean_text.strip():
            candidates.append(
                {
                    "text": clean_text,
                    "start_token": start_index,
                    "end_token": len(tokens),
                }
            )

    return candidates


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Standard normalization for SQuAD-style evaluation.
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


def f1_score(prediction, ground_truth):
    """Computes F1 score between prediction and ground truth strings."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Computes exact match score (0 or 1) between prediction and ground truth."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Computes the maximum metric score over multiple ground truths.

    Args:
        metric_fn (function): The metric function (f1_score or exact_match_score).
        prediction (str): The predicted answer string.
        ground_truths (list): A list of valid ground truth answer strings.

    Returns:
        float: The maximum score.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0


def compute_f1(prediction, ground_truths):
    """Computes max F1 score for a prediction against a list of ground truths."""
    return metric_max_over_ground_truths(f1_score, prediction, ground_truths)


def exact_match(prediction, ground_truths):
    """Computes max Exact Match score for a prediction against a list of ground truths."""
    return metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
