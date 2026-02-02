import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class SmartBatchingCollate:
    """
    Collate function that applies dynamic padding to the batch.
    It pads input_ids, attention_mask, and token_type_ids to the maximum sequence length
    present in the current batch, rather than a fixed global max_len.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # batch is a list of dictionaries

        # Extract sequences
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]
        offsets = [item["offsets"] for item in batch]

        # Determine max length in this batch
        max_len = max(len(ids) for ids in input_ids)

        # Prepare output tensors
        batch_size = len(batch)

        # Initialize tensors with padding values
        padded_input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        padded_attention_mask = torch.full((batch_size, max_len), 0, dtype=torch.long)
        padded_token_type_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
        padded_offsets = torch.full((batch_size, max_len, 2), 0, dtype=torch.long)

        # Stack scalar targets
        start_targets = torch.tensor(
            [item["start_target"] for item in batch], dtype=torch.long
        )
        end_targets = torch.tensor(
            [item["end_target"] for item in batch], dtype=torch.long
        )

        # Fill tensors
        for i in range(batch_size):
            length = len(input_ids[i])
            padded_input_ids[i, :length] = torch.tensor(input_ids[i], dtype=torch.long)
            padded_attention_mask[i, :length] = torch.tensor(
                attention_mask[i], dtype=torch.long
            )
            padded_token_type_ids[i, :length] = torch.tensor(
                token_type_ids[i], dtype=torch.long
            )
            padded_offsets[i, :length] = torch.tensor(offsets[i], dtype=torch.long)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "token_type_ids": padded_token_type_ids,
            "start_targets": start_targets,
            "end_targets": end_targets,
            "offsets": padded_offsets,
            "text": [item["text"] for item in batch],
            "sentiment": [item["sentiment"] for item in batch],
            "selected_text": [item["selected_text"] for item in batch],
            "textID": [item["textID"] for item in batch],
        }


class TweetDataset(Dataset):
    """
    Dataset class that wraps pre-processed numpy arrays.
    """

    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]
        self.offsets = data["offsets"]
        self.start_targets = data["start_targets"]
        self.end_targets = data["end_targets"]

        # Metadata
        self.texts = data["texts"]
        self.sentiments = data["sentiments"]
        self.selected_texts = data["selected_texts"]
        self.text_ids = data["text_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "offsets": self.offsets[idx],
            "start_target": self.start_targets[idx],
            "end_target": self.end_targets[idx],
            "text": self.texts[idx],
            "sentiment": self.sentiments[idx],
            "selected_text": self.selected_texts[idx],
            "textID": self.text_ids[idx],
        }


def get_processed_data(
    df, tokenizer, max_len, model_name, stage="train", load_cached_data=True
):
    """
    Processes the dataframe into tokenized features and targets.
    Implements caching to disk to speed up subsequent runs.
    """
    # Create cache directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_")
    cache_path = os.path.join(
        Config.OUTPUT_DIR, f"cached_{stage}_{safe_model_name}.npz"
    )

    # Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"],
                "token_type_ids": data["token_type_ids"],
                "offsets": data["offsets"],
                "start_targets": data["start_targets"],
                "end_targets": data["end_targets"],
                "texts": data["texts"],
                "sentiments": data["sentiments"],
                "selected_texts": data["selected_texts"],
                "text_ids": data["text_ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data for {model_name} ({stage})...")

    # Lists to store processed data
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    offsets_list = []
    start_targets = []
    end_targets = []

    # Metadata lists
    texts = df["text"].values
    sentiments = df["sentiment"].values
    text_ids = df["textID"].values

    # Handle selected_text (only available in train/val)
    if "selected_text" in df.columns:
        selected_texts = df["selected_text"].values
    else:
        selected_texts = [""] * len(df)  # Dummy for test set

    for i in range(len(df)):
        text = str(texts[i])
        sentiment = str(sentiments[i])
        selected_text = str(selected_texts[i])

        # Tokenize: [CLS] Sentiment [SEP] Text [SEP] (or similar based on tokenizer)
        # We put sentiment first to condition the attention
        # return_offsets_mapping=True is crucial for finding targets
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]
        offsets = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        # Find targets
        start_idx = 0
        end_idx = 0

        if stage != "test":
            # Find the start and end character positions of selected_text in text
            # Note: selected_text is a substring of text.
            # We search for the substring.
            # Sometimes there are multiple occurrences; usually the first one is intended.
            # Also, there might be slight whitespace discrepancies.

            # Strict find
            char_start = text.find(selected_text)
            if char_start == -1:
                # Fallback: if not found (rare), assume full text
                char_start = 0
                char_end = len(text)
            else:
                char_end = char_start + len(selected_text)

            # Map character positions to token indices
            # We only look at tokens belonging to the 'text' part (sequence_id == 1)
            token_start_index = 0
            token_end_index = 0
            found_start = False
            found_end = False

            for idx, (seq_id, offset) in enumerate(zip(sequence_ids, offsets)):
                # Skip special tokens and sentiment tokens
                if seq_id != 1:
                    continue

                # Check start
                # If the token contains the start character or starts after it (but is the first one)
                # Ideally, the token containing the start char.
                if (
                    not found_start
                    and offset[0] <= char_start
                    and offset[1] > char_start
                ):
                    token_start_index = idx
                    found_start = True

                # Check end
                # The token containing the last character of selected_text
                # char_end points to index after last char. So last char is char_end - 1
                if offset[0] <= (char_end - 1) and offset[1] > (char_end - 1):
                    token_end_index = idx
                    found_end = True

            # Fallback if exact token overlap not found (e.g. weird spacing)
            if not found_start:
                # Find first token of sequence 1
                for idx, seq_id in enumerate(sequence_ids):
                    if seq_id == 1:
                        token_start_index = idx
                        break

            if not found_end:
                # Find last token of sequence 1
                for idx, seq_id in enumerate(sequence_ids):
                    if seq_id == 1:
                        token_end_index = idx
                # If still 0 (empty text?), set to start
                if token_end_index < token_start_index:
                    token_end_index = token_start_index

            start_idx = token_start_index
            end_idx = token_end_index

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        offsets_list.append(offsets)
        start_targets.append(start_idx)
        end_targets.append(end_idx)

    # Convert to numpy object arrays (ragged) for lists, standard arrays for scalars
    # We keep them as object arrays of lists because lengths vary, and we pad in Collate
    data_dict = {
        "input_ids": np.array(input_ids_list, dtype=object),
        "attention_mask": np.array(attention_mask_list, dtype=object),
        "token_type_ids": np.array(token_type_ids_list, dtype=object),
        "offsets": np.array(offsets_list, dtype=object),
        "start_targets": np.array(start_targets, dtype=np.int64),
        "end_targets": np.array(end_targets, dtype=np.int64),
        "texts": np.array(texts, dtype=object),
        "sentiments": np.array(sentiments, dtype=object),
        "selected_texts": np.array(selected_texts, dtype=object),
        "text_ids": np.array(text_ids, dtype=object),
    }

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    np.savez(cache_path, **data_dict)

    return data_dict


def get_dataloaders(model_config, load_cached_data=True, debug=False):
    """
    Main function to create DataLoaders for train, validation, and test sets.

    Args:
        model_config (dict): Configuration dictionary for the specific model (from Config.MODEL_CONFIGS).
        load_cached_data (bool): Whether to load pre-processed data from cache.
        debug (bool): If True, uses a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.SEED)

    model_name = model_config["model_name"]
    batch_size = model_config["batch_size"]
    tokenizer_type = model_config.get("tokenizer_type", "roberta")

    print(f"\nInitializing DataLoaders for {model_name}...")

    # Initialize Tokenizer
    # use_fast=True is required for offset_mapping and sequence_ids
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load Metadata Dataframes
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Debug Mode
    if debug:
        print(f"DEBUG mode: using {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Pre-processing: Sort training data by length for Smart Batching
    # This groups similar length sequences together, minimizing padding in batches
    print("Sorting training data by text length for smart batching...")
    df_train["text_len"] = df_train["text"].astype(str).apply(len)
    df_train = df_train.sort_values("text_len").reset_index(drop=True)
    # Drop the temporary column
    df_train = df_train.drop(columns=["text_len"])

    # Process Data
    train_data = get_processed_data(
        df_train, tokenizer, Config.MAX_LEN, model_name, "train", load_cached_data
    )
    val_data = get_processed_data(
        df_val, tokenizer, Config.MAX_LEN, model_name, "val", load_cached_data
    )
    test_data = get_processed_data(
        df_test, tokenizer, Config.MAX_LEN, model_name, "test", load_cached_data
    )

    # Create Datasets
    train_dataset = TweetDataset(train_data)
    val_dataset = TweetDataset(val_data)
    test_dataset = TweetDataset(test_data)

    # Create Collate Function
    collate_fn = SmartBatchingCollate(pad_token_id=tokenizer.pad_token_id)

    # Create DataLoaders
    # Note: shuffle=False for train_loader because we already sorted it for smart batching.
    # If we shuffle, we lose the smart batching benefit.
    # However, to avoid identical batches every epoch, one might shuffle blocks,
    # but simple sorting is standard for this specific optimization in limited time.
    # Actually, strictly sorted data can bias BN/optimization.
    # A common compromise is to sort, then use a BatchSampler, or just shuffle=False for short training.
    # Given the prompt "Sort the training data... construct batches from samples of similar lengths",
    # shuffle=False on the sorted dataset is the intended implementation.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
