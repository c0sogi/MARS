import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer
from library.configuration import Config
from library.utilities import set_seed


# --------------------------------------------------------------------------
# Dataset Class
# --------------------------------------------------------------------------
class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering.
    Wraps the processed features (input_ids, attention_mask, labels).
    """

    def __init__(self, features_df, is_train=True):
        self.is_train = is_train
        self.data = features_df.reset_index(drop=True)

        # Convert list columns from pandas (which might be numpy arrays or lists) to appropriate formats
        # We assume the dataframe columns contain lists of integers
        self.input_ids = self.data["input_ids"].tolist()
        self.attention_mask = self.data["attention_mask"].tolist()

        if self.is_train:
            self.labels = self.data["labels"].tolist()
            # Flag for the sampler to identify positive samples
            self.is_positive = self.data["is_positive"].values
        else:
            self.example_ids = self.data["example_id"].tolist()
            # offset_mapping is a list of lists/tuples
            self.offset_mapping = self.data["offset_mapping"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.is_train:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # For inference, we need metadata to map back to original text
            item["example_id"] = self.example_ids[idx]
            # Offset mapping is returned as a tensor or list, but usually collate_fn handles lists better
            # Here we return it as a tensor if possible, but offsets are tuples.
            # We will return it as a python object and handle it in a custom collate or just loop over dataset
            # To be safe for standard DataLoaders, we won't tensor-ize offset_mapping here if it's complex.
            # However, for the model forward pass, we don't need offset_mapping.
            # We need it for post-processing.
            item["offset_mapping"] = torch.tensor(
                self.offset_mapping[idx], dtype=torch.long
            )

        return item


# --------------------------------------------------------------------------
# Custom Sampler
# --------------------------------------------------------------------------
class PositiveAnchoredSampler(Sampler):
    """
    Stratified Batch Sampler.
    Constructs batches such that every batch contains at least one positive window
    (a window containing the answer) to prevent gradient starvation.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        if not hasattr(data_source, "is_positive"):
            raise ValueError(
                "DataSource must have 'is_positive' attribute for PositiveAnchoredSampler"
            )

        # Separate indices
        self.pos_indices = np.where(data_source.is_positive)[0].tolist()
        self.neg_indices = np.where(~data_source.is_positive)[0].tolist()

        # Calculate expected length (approximate)
        # We primarily want to cover all negatives (majority), anchored by positives.
        # Each batch consumes (batch_size - 1) negatives.
        if len(self.neg_indices) > 0:
            self.num_batches = int(
                np.ceil(len(self.neg_indices) / max(1, self.batch_size - 1))
            )
        else:
            self.num_batches = int(np.ceil(len(self.pos_indices) / self.batch_size))

    def __iter__(self):
        # Shuffle indices for this epoch
        pos = np.random.permutation(self.pos_indices).tolist()
        neg = np.random.permutation(self.neg_indices).tolist()

        # We need 'num_batches' positives to anchor the batches.
        # If we have fewer positives than batches, we cycle them (oversampling positives).
        # If we have more, we use them all (some batches might have >1 positive, or we just use the surplus).

        # Strategy: Ensure at least 1 pos per batch.
        # 1. Prepare list of positives for the batches
        anchors = []
        while len(anchors) < self.num_batches:
            anchors.extend(pos)
        anchors = anchors[: self.num_batches]  # Trim to exact number needed

        batches = []
        neg_ptr = 0

        for i in range(self.num_batches):
            batch = []
            # 1. Add Anchor
            batch.append(anchors[i])

            # 2. Fill with Negatives
            slots_needed = self.batch_size - 1
            if slots_needed > 0:
                end_ptr = min(neg_ptr + slots_needed, len(neg))
                batch.extend(neg[neg_ptr:end_ptr])
                neg_ptr = end_ptr

                # If we ran out of negatives but batch is not full, fill with random samples (pos or neg)
                # to maintain batch size stability (optional but good for compute efficiency)
                while len(batch) < self.batch_size:
                    # Pick random index from all data
                    rand_idx = np.random.randint(0, len(self.data_source))
                    if rand_idx not in batch:
                        batch.append(rand_idx)
                    else:
                        # Simple fallback
                        break

            batches.append(batch)

        # Shuffle the order of batches
        np.random.shuffle(batches)

        # Flatten
        final_indices = [idx for batch in batches for idx in batch]
        return iter(final_indices)

    def __len__(self):
        return self.num_batches * self.batch_size


# --------------------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------------------
def prepare_train_features(examples, tokenizer):
    """
    Tokenizes examples with sliding window and generates Soft Overlap BIO labels.
    """
    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Storage for processed features
    features = {"input_ids": [], "attention_mask": [], "labels": [], "is_positive": []}

    # Labels: 0=O, 1=B-ANS, 2=I-ANS

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]

        # Get ground truth details
        # Note: examples is a DataFrame slice, so we access by integer index via iloc if needed,
        # but here 'examples' is passed as a dataframe, so we use iloc with sample_index
        answer_text = examples.iloc[sample_index]["answer_text"]
        start_char = examples.iloc[sample_index]["answer_start"]
        end_char = start_char + len(answer_text)

        # Initialize labels as O (0)
        labels = [0] * len(input_ids)
        has_answer = False

        # Find context token indices
        # sequence_ids: 0 for question, 1 for context, None for special tokens
        token_indices = [idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1]

        if not token_indices:
            # Should not happen with valid context
            features["input_ids"].append(input_ids)
            features["attention_mask"].append(attention_mask)
            features["labels"].append(labels)
            features["is_positive"].append(False)
            continue

        # Soft Overlap Labeling
        # Iterate through context tokens and check overlap
        first_overlap = True

        for idx in token_indices:
            # offsets[idx] is a tuple (start, end) of the token in the original string
            token_start, token_end = offsets[idx]

            # Skip padding or empty tokens
            if token_start == token_end:
                continue

            # Check overlap: token range [token_start, token_end) vs answer range [start_char, end_char)
            # Overlap condition: max(token_start, start_char) < min(token_end, end_char)
            if max(token_start, start_char) < min(token_end, end_char):
                has_answer = True
                if first_overlap:
                    labels[idx] = 1  # B-ANS
                    first_overlap = False
                else:
                    labels[idx] = 2  # I-ANS
            else:
                labels[idx] = 0  # O

        # Mask labels for special tokens / question tokens (set to -100 to ignore in loss if using CrossEntropy)
        # However, we initialized to 0 (O).
        # For token classification, usually we set ignore_index=-100.
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id != 1:
                labels[idx] = -100

        features["input_ids"].append(input_ids)
        features["attention_mask"].append(attention_mask)
        features["labels"].append(labels)
        features["is_positive"].append(has_answer)

    return pd.DataFrame(features)


def prepare_eval_features(examples, tokenizer):
    """
    Tokenizes examples with sliding window for evaluation/inference.
    Retains metadata for post-processing.
    """
    tokenized_examples = tokenizer(
        examples["question"].tolist(),
        examples["context"].tolist(),
        truncation="only_second",
        max_length=Config.MAX_LENGTH,
        stride=Config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]  # Keep it

    features = {
        "input_ids": [],
        "attention_mask": [],
        "example_id": [],
        "offset_mapping": [],
    }

    for i in range(len(tokenized_examples["input_ids"])):
        features["input_ids"].append(tokenized_examples["input_ids"][i])
        features["attention_mask"].append(tokenized_examples["attention_mask"][i])

        sample_index = sample_mapping[i]
        features["example_id"].append(examples.iloc[sample_index]["id"])

        # We need to store offset_mapping. It is a list of tuples.
        # We convert tuples to lists to ensure compatibility with parquet/pandas serialization if needed
        # though pandas handles tuples in object columns fine usually.
        features["offset_mapping"].append([list(o) for o in offset_mapping[i]])

    return pd.DataFrame(features)


# --------------------------------------------------------------------------
# Main Data Loading Function
# --------------------------------------------------------------------------
def get_qa_data(load_cached_data=True):
    """
    Main entry point to get Train, Val, and Test datasets.
    Handles caching to disk to save time on re-runs.

    Args:
        load_cached_data (bool): Whether to try loading from parquet cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    set_seed(Config.SEED)

    # Define Cache Paths
    os.makedirs(Config.QA_CACHE_DIR, exist_ok=True)
    train_cache = os.path.join(Config.QA_CACHE_DIR, "train_features.parquet")
    val_cache = os.path.join(Config.QA_CACHE_DIR, "val_features.parquet")
    test_cache = os.path.join(Config.QA_CACHE_DIR, "test_features.parquet")

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading QA features from cache: {Config.QA_CACHE_DIR}")
        try:
            df_train_feats = pd.read_parquet(train_cache)
            df_val_feats = pd.read_parquet(val_cache)
            df_test_feats = pd.read_parquet(test_cache)

            # Reconstruct Datasets
            train_ds = QADataset(df_train_feats, is_train=True)
            val_ds = QADataset(
                df_val_feats, is_train=True
            )  # Val has labels for evaluation
            test_ds = QADataset(df_test_feats, is_train=False)

            return train_ds, val_ds, test_ds
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print("Processing QA data from metadata...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_FILE)
    df_val = pd.read_csv(Config.VAL_FILE)
    df_test = pd.read_csv(Config.TEST_FILE)

    # Debugging limits
    if Config.MAX_TRAIN_SAMPLES:
        df_train = df_train.iloc[: Config.MAX_TRAIN_SAMPLES]
        df_test = df_test.iloc[
            : Config.MAX_TRAIN_SAMPLES
        ]  # Just for consistency in debug
    if Config.MAX_VAL_SAMPLES:
        df_val = df_val.iloc[: Config.MAX_VAL_SAMPLES]

    # Load Tokenizer
    # Use the TAPT-finetuned tokenizer if available, else base
    if os.path.exists(Config.TAPT_OUTPUT_DIR):
        print(f"Using TAPT tokenizer from {Config.TAPT_OUTPUT_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(Config.TAPT_OUTPUT_DIR)
    else:
        print(f"Using base tokenizer {Config.MODEL_CHECKPOINT}")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # Process Data
    print("Generating Train features...")
    df_train_feats = prepare_train_features(df_train, tokenizer)

    print("Generating Val features...")
    df_val_feats = prepare_train_features(
        df_val, tokenizer
    )  # Val also needs labels for loss calc

    print("Generating Test features...")
    df_test_feats = prepare_eval_features(df_test, tokenizer)

    # Save to Cache
    print(f"Saving features to {Config.QA_CACHE_DIR}")
    df_train_feats.to_parquet(train_cache, index=False)
    df_val_feats.to_parquet(val_cache, index=False)
    df_test_feats.to_parquet(test_cache, index=False)

    # Create Datasets
    train_ds = QADataset(df_train_feats, is_train=True)
    val_ds = QADataset(df_val_feats, is_train=True)
    test_ds = QADataset(df_test_feats, is_train=False)

    return train_ds, val_ds, test_ds
