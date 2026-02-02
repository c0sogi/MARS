import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, logging as hf_logging
from library.config import cfg

# Suppress Transformers logging to keep output clean
hf_logging.set_verbosity_error()

# Global variables to cache tokenizers in memory
_ROUTER_TOKENIZER = None
_GENERATOR_TOKENIZER = None


def get_router_tokenizer():
    """Returns the cached Router tokenizer (DeBERTa)."""
    global _ROUTER_TOKENIZER
    if _ROUTER_TOKENIZER is None:
        _ROUTER_TOKENIZER = AutoTokenizer.from_pretrained(
            cfg.ROUTER_MODEL_NAME, use_fast=True
        )
    return _ROUTER_TOKENIZER


def get_generator_tokenizer():
    """Returns the cached Generator tokenizer (ByT5)."""
    global _GENERATOR_TOKENIZER
    if _GENERATOR_TOKENIZER is None:
        _GENERATOR_TOKENIZER = AutoTokenizer.from_pretrained(
            cfg.GENERATOR_MODEL_NAME, use_fast=True
        )
    return _GENERATOR_TOKENIZER


class TextNormalizationDataset(Dataset):
    """Generic Dataset wrapper for tokenized inputs and optional labels."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert list/array items to tensors on demand
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def process_router_data(df, tokenizer, is_train=True):
    """
    Processes raw dataframe into tokenized inputs for the Router (Token Classification).
    Handles grouping by sentence, downsampling PLAIN sentences, and label alignment.
    """
    grouped = df.groupby("sentence_id")
    sentences = []
    sentence_labels = []

    # Iterate over sentences
    for _, group in grouped:
        words = group["before"].astype(str).tolist()

        if "class" in group.columns:
            classes = group["class"].tolist()

            if is_train:
                # Strategic Sampling: Keep all sentences with non-PLAIN tokens.
                # Downsample sentences that are purely PLAIN.
                has_interesting = any(c != "PLAIN" for c in classes)
                if not has_interesting:
                    if np.random.rand() > cfg.PLAIN_DOWNSAMPLE_RATIO:
                        continue

            # Map class names to IDs
            label_ids = [cfg.CLASS2ID.get(c, 0) for c in classes]
            sentence_labels.append(label_ids)
        else:
            # Test set case (no labels)
            sentence_labels.append(None)

        sentences.append(words)

    # Tokenize sentences (Pre-tokenized input)
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=cfg.MAX_LENGTH_ROUTER,
        return_tensors=None,  # Return lists for easier caching/manipulation
    )

    # Align labels with subword tokens
    final_labels = []
    if sentence_labels and sentence_labels[0] is not None:
        for i, label in enumerate(sentence_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a token gets the label
                    label_ids.append(label[word_idx])
                else:
                    # Subsequent subwords get ignored
                    label_ids.append(-100)
                previous_word_idx = word_idx
            final_labels.append(label_ids)
    else:
        final_labels = None

    return tokenized_inputs, final_labels


def load_router_data(split="train", load_cached_data=True):
    """
    Loads or creates the Router dataset for a specific split.
    Implements caching to disk.
    """
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.CACHE_DIR, f"router_{split}_processed.pt")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached router data for {split}...")
        data = torch.load(cache_path)
        return TextNormalizationDataset(data["encodings"], data["labels"])

    print(f"Processing router data for {split}...")
    if split == "train":
        df = pd.read_csv(cfg.TRAIN_FILE, keep_default_na=False)
        is_train = True
    elif split == "val":
        df = pd.read_csv(cfg.VAL_FILE, keep_default_na=False)
        is_train = False
    else:
        raise ValueError("Invalid split. Must be 'train' or 'val'.")

    tokenizer = get_router_tokenizer()
    encodings, labels = process_router_data(df, tokenizer, is_train=is_train)

    # Save to cache
    torch.save({"encodings": encodings, "labels": labels}, cache_path)
    return TextNormalizationDataset(encodings, labels)


def process_generator_data(df, tokenizer):
    """
    Processes raw dataframe into context-augmented inputs for the Generator (Seq2Seq).
    Filters for NEURAL_BASED_CLASSES and constructs windowed inputs.
    """
    grouped = df.groupby("sentence_id")
    input_texts = []
    target_texts = []

    for _, group in grouped:
        words = group["before"].astype(str).tolist()
        classes = group["class"].tolist()
        afters = group["after"].astype(str).tolist()

        for i, (cls, word, target) in enumerate(zip(classes, words, afters)):
            if cls in cfg.NEURAL_BASED_CLASSES:
                # Construct Context Window
                start = max(0, i - cfg.CONTEXT_WINDOW)
                end = min(len(words), i + cfg.CONTEXT_WINDOW + 1)

                left_ctx = words[start:i]
                right_ctx = words[i + 1 : end]

                # Format: [CLASS] left <extra_id_0> target <extra_id_1> right
                input_str = f"[{cls}] {' '.join(left_ctx)} <extra_id_0> {word} <extra_id_1> {' '.join(right_ctx)}"

                input_texts.append(input_str)
                target_texts.append(target)

    # Tokenize inputs
    model_inputs = tokenizer(
        input_texts,
        max_length=cfg.MAX_LENGTH_GENERATOR,
        padding=True,
        truncation=True,
        return_tensors=None,
    )

    # Tokenize targets
    # Use text_target for modern transformers, or fallback to standard call
    try:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_texts,
                max_length=cfg.MAX_LENGTH_GENERATOR,
                padding=True,
                truncation=True,
                return_tensors=None,
            )
    except:
        labels = tokenizer(
            text_target=target_texts,
            max_length=cfg.MAX_LENGTH_GENERATOR,
            padding=True,
            truncation=True,
            return_tensors=None,
        )

    return model_inputs, labels["input_ids"]


def load_generator_data(split="train", load_cached_data=True):
    """
    Loads or creates the Generator dataset for a specific split.
    Implements caching to disk.
    """
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.CACHE_DIR, f"generator_{split}_processed.pt")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached generator data for {split}...")
        data = torch.load(cache_path)
        return TextNormalizationDataset(data["encodings"], data["labels"])

    print(f"Processing generator data for {split}...")
    if split == "train":
        df = pd.read_csv(cfg.TRAIN_FILE, keep_default_na=False)
    elif split == "val":
        df = pd.read_csv(cfg.VAL_FILE, keep_default_na=False)
    else:
        raise ValueError("Invalid split. Must be 'train' or 'val'.")

    tokenizer = get_generator_tokenizer()
    encodings, labels = process_generator_data(df, tokenizer)

    torch.save({"encodings": encodings, "labels": labels}, cache_path)
    return TextNormalizationDataset(encodings, labels)


def load_test_router_data():
    """
    Loads and processes the test set for the Router model.
    Returns the dataset and the raw dataframe (for ID mapping).
    """
    print("Processing test data for Router...")
    df = pd.read_csv(cfg.TEST_FILE, keep_default_na=False)
    tokenizer = get_router_tokenizer()

    # Reuse process_router_data logic (handles missing 'class' column)
    encodings, _ = process_router_data(df, tokenizer, is_train=False)

    return TextNormalizationDataset(encodings, None), df


def prepare_generator_inference_data(df_test, predicted_classes):
    """
    Prepares inputs for the Generator based on Router predictions on the test set.

    Args:
        df_test (pd.DataFrame): The test dataframe.
        predicted_classes (list of lists): Predicted class IDs for each sentence.

    Returns:
        dataset (TextNormalizationDataset): Dataset containing generator inputs.
        metadata (list): List of (sentence_id, token_id) tuples corresponding to inputs.
    """
    print("Preparing generator inference data...")
    tokenizer = get_generator_tokenizer()

    # Ensure we iterate in the same order as the predictions (grouped by sentence_id)
    sent_ids = sorted(df_test["sentence_id"].unique())

    input_texts = []
    metadata = []  # Stores (sentence_id, token_id) to map predictions back

    if len(sent_ids) != len(predicted_classes):
        print(
            f"Warning: Mismatch in sentence count. Data: {len(sent_ids)}, Preds: {len(predicted_classes)}"
        )

    for idx, sent_id in enumerate(sent_ids):
        group = df_test[df_test["sentence_id"] == sent_id]
        words = group["before"].astype(str).tolist()
        token_ids = group["token_id"].tolist()

        # Get predictions for this sentence
        pred_ids = predicted_classes[idx]

        # Safety alignment
        min_len = min(len(pred_ids), len(words))

        for i in range(min_len):
            p_id = pred_ids[i]
            word = words[i]
            t_id = token_ids[i]

            cls_name = cfg.ID2CLASS[p_id]

            if cls_name in cfg.NEURAL_BASED_CLASSES:
                # Construct Context
                start = max(0, i - cfg.CONTEXT_WINDOW)
                end = min(len(words), i + cfg.CONTEXT_WINDOW + 1)
                left_ctx = words[start:i]
                right_ctx = words[i + 1 : end]

                input_str = f"[{cls_name}] {' '.join(left_ctx)} <extra_id_0> {word} <extra_id_1> {' '.join(right_ctx)}"

                input_texts.append(input_str)
                metadata.append((sent_id, t_id))

    if not input_texts:
        return None, []

    model_inputs = tokenizer(
        input_texts,
        max_length=cfg.MAX_LENGTH_GENERATOR,
        padding=True,
        truncation=True,
        return_tensors="pt",  # Return tensors directly here or lists? Dataset expects dict of tensors/lists.
    )

    # Convert BatchEncoding to dict of lists/tensors for Dataset
    encodings = {k: v for k, v in model_inputs.items()}

    return TextNormalizationDataset(encodings, None), metadata
