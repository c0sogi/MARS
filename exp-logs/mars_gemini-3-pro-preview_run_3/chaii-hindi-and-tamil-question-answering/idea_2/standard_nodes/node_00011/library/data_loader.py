import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering with BIO tagging.
    """

    def __init__(self, data_df):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing processed features.
        """
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Helper to convert potential numpy object arrays to lists
        def ensure_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        # Convert lists to tensors
        # Note: Parquet stores lists as numpy arrays or lists, we ensure conversion to tensor
        return {
            "input_ids": torch.tensor(ensure_list(row["input_ids"]), dtype=torch.long),
            "attention_mask": torch.tensor(
                ensure_list(row["attention_mask"]), dtype=torch.long
            ),
            "labels": torch.tensor(ensure_list(row["labels"]), dtype=torch.long),
            "offset_mapping": torch.tensor(
                ensure_list(row["offset_mapping"]), dtype=torch.long
            ),
            "example_id": str(row["example_id"]),
        }


def process_data(df, tokenizer, mode="train"):
    """
    Tokenizes data with sliding window and generates BIO labels.

    Args:
        df (pd.DataFrame): Raw metadata dataframe.
        tokenizer: Transformers tokenizer.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: Processed features.
    """
    # Prepare lists for batch processing
    questions = df["question"].astype(str).tolist()
    contexts = df["context"].astype(str).tolist()

    # Tokenize with sliding window
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=Config.MAX_LEN,
        stride=Config.STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []

    # Iterate over each window (feature)
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        example_id = df.iloc[sample_index]["id"]

        # Initialize labels with -100 (ignore)
        labels = [-100] * len(input_ids)

        if mode != "test":
            answer_start = df.iloc[sample_index]["answer_start"]
            answer_text = df.iloc[sample_index]["answer_text"]
            # Calculate answer end character index
            answer_end = answer_start + len(answer_text)

            # Identify context tokens
            # sequence_ids: 0 for question, 1 for context, None for special tokens
            token_indices = [
                idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if token_indices:
                ctx_start_idx = token_indices[0]
                ctx_end_idx = token_indices[-1]

                # Get character span of the context in this window
                window_start_char = offsets[ctx_start_idx][0]
                window_end_char = offsets[ctx_end_idx][1]

                # Check if the answer is fully contained in this window
                if window_start_char <= answer_start and window_end_char >= answer_end:
                    # Initialize context labels to O (0)
                    for idx in token_indices:
                        labels[idx] = 0

                    # Find start token index
                    start_token = ctx_start_idx
                    while (
                        start_token <= ctx_end_idx
                        and offsets[start_token][0] <= answer_start
                    ):
                        start_token += 1
                    start_token -= 1

                    # Find end token index
                    end_token = ctx_end_idx
                    while (
                        end_token >= ctx_start_idx
                        and offsets[end_token][1] >= answer_end
                    ):
                        end_token -= 1
                    end_token += 1

                    # Assign BIO labels
                    if start_token <= end_token:
                        labels[start_token] = Config.LABEL2ID["B-ANS"]
                        for k in range(start_token + 1, end_token + 1):
                            labels[k] = Config.LABEL2ID["I-ANS"]
                else:
                    # Answer not in window or partial -> Label context as O
                    for idx in token_indices:
                        labels[idx] = 0
        else:
            # Test mode: No labels needed, but we set context to 0 (O) or -100
            # Setting to -100 as we don't calculate loss in inference usually,
            # but consistent shape is good.
            pass

        features.append(
            {
                "example_id": example_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "offset_mapping": offsets,
                "has_answer": Config.LABEL2ID["B-ANS"] in labels,
            }
        )

    return pd.DataFrame(features)


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Loads data, processes it (with caching), and returns DataLoaders.

    Args:
        debug (bool): If True, use a small subset of data.
        load_cached_data (bool): If True, attempt to load from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(42)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    loaders = {}
    modes = ["train", "val", "test"]
    files = {
        "train": Config.TRAIN_FILE,
        "val": Config.VAL_FILE,
        "test": Config.TEST_FILE,
    }

    for mode in modes:
        cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_cache_v2.parquet")

        # 1. Try Loading Cache
        df_processed = None
        if load_cached_data and os.path.exists(cache_path):
            try:
                # print(f"Loading {mode} data from cache: {cache_path}")
                df_processed = pd.read_parquet(cache_path)
            except Exception:
                # print(f"Failed to load cache for {mode}, reprocessing...")
                pass

        # 2. Process if needed
        if df_processed is None:
            # print(f"Processing {mode} data...")
            df_raw = pd.read_csv(files[mode])

            if debug:
                df_raw = df_raw.head(20)

            df_processed = process_data(df_raw, tokenizer, mode=mode)

            if mode == "train":
                # Downsample negatives to mitigate label sparsity (Cite solution_lesson_node_00010)
                pos_mask = df_processed["has_answer"] == True
                neg_mask = df_processed["has_answer"] == False

                df_pos = df_processed[pos_mask]
                df_neg = df_processed[neg_mask]

                # Keep 20% of negatives
                df_neg_sampled = df_neg.sample(frac=0.2, random_state=42)

                df_processed = (
                    pd.concat([df_pos, df_neg_sampled])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )

            # Save to cache
            # print(f"Saving {mode} data to cache...")
            df_processed.to_parquet(cache_path, index=False)

        # 3. Create Dataset and DataLoader
        dataset = QADataset(df_processed)

        shuffle = mode == "train"
        loaders[mode] = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
