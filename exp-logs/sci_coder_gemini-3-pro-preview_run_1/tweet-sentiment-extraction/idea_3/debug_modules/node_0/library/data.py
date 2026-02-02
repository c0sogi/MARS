import os
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import TweetConfig
from library.utils import seed_everything

# Suppress verbose tokenizer warnings
transformers.logging.set_verbosity_error()


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Returns input IDs, attention masks, soft targets for start/end indices,
    offsets, and original text/sentiment for post-processing.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_tokens,
        end_tokens,
        offsets,
        texts,
        sentiments,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.offsets = offsets
        self.texts = texts
        self.sentiments = sentiments

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "start_tokens": torch.tensor(self.start_tokens[item], dtype=torch.float),
            "end_tokens": torch.tensor(self.end_tokens[item], dtype=torch.float),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
            "text": str(self.texts[item]),
            "sentiment": str(self.sentiments[item]),
        }


def normalize_text(text):
    """
    Applies strict whitespace normalization to ensure alignment.
    Collapses multiple spaces into one and strips leading/trailing whitespace.
    """
    return " ".join(str(text).split())


def process_data(df, tokenizer, config, mode="train"):
    """
    Processes the dataframe: normalizes text, tokenizes, and generates Gaussian targets.
    """
    input_ids_list = []
    attention_mask_list = []
    start_tokens_list = []
    end_tokens_list = []
    offsets_list = []
    texts_list = []
    sentiments_list = []

    # Iterate over the dataframe
    for _, row in df.iterrows():
        text = normalize_text(row["text"])
        sentiment = row["sentiment"]

        # Determine selected_text
        selected_text = None
        if mode != "test" and "selected_text" in row:
            selected_text = normalize_text(row["selected_text"])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # We pass sentiment as the first sequence and text as the second (pair)
        # This allows us to easily identify text tokens via sequence_ids
        encoding = tokenizer(
            sentiment,
            text,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]
        sequence_ids = encoding.sequence_ids()

        # Initialize targets (soft labels)
        start_target = np.zeros(config.MAX_LEN, dtype=np.float32)
        end_target = np.zeros(config.MAX_LEN, dtype=np.float32)

        if selected_text:
            # Find character indices of selected_text in the normalized text
            start_idx = text.find(selected_text)

            # Handle edge case where normalization might cause mismatch (rare)
            if start_idx == -1:
                start_idx = 0
                end_idx = len(text)
            else:
                end_idx = start_idx + len(selected_text)

            # Identify tokens belonging to the 'text' part (sequence_id == 1)
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if text_token_indices:
                token_start_index = text_token_indices[0]
                token_end_index = text_token_indices[-1]

                # Map character indices to token indices
                for i in text_token_indices:
                    token_offsets = offsets[i]
                    # Check if token contains the start character
                    if token_offsets[0] <= start_idx < token_offsets[1]:
                        token_start_index = i
                    # Check if token contains the last character of the selection
                    # (end_idx is exclusive, so we check end_idx - 1)
                    if token_offsets[0] <= (end_idx - 1) < token_offsets[1]:
                        token_end_index = i

                # Ensure validity
                if token_start_index > token_end_index:
                    token_end_index = token_start_index

                # Generate Gaussian-smoothed targets
                indices = np.arange(config.MAX_LEN)

                # Start distribution
                start_dist = np.exp(
                    -0.5 * ((indices - token_start_index) / config.SIGMA) ** 2
                )
                start_target = start_dist / start_dist.sum()

                # End distribution
                end_dist = np.exp(
                    -0.5 * ((indices - token_end_index) / config.SIGMA) ** 2
                )
                end_target = end_dist / end_dist.sum()

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        start_tokens_list.append(start_target)
        end_tokens_list.append(end_target)
        offsets_list.append(offsets)
        texts_list.append(text)
        sentiments_list.append(sentiment)

    return (
        np.array(input_ids_list),
        np.array(attention_mask_list),
        np.array(start_tokens_list),
        np.array(end_tokens_list),
        np.array(offsets_list),
        np.array(texts_list),
        np.array(sentiments_list),
    )


def get_loaders(load_cached_data=True):
    """
    Loads data, performs filtering/processing (with caching), and returns DataLoaders.
    Strictly excludes 'neutral' tweets from training and validation.
    """
    config = TweetConfig()
    seed_everything(config.SEED)

    # Ensure cache directory exists
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    data_arrays = {}

    # Check cache existence
    missing_cache = False
    if load_cached_data:
        for split in splits:
            # Check for numeric arrays
            for key in [
                "input_ids",
                "attention_mask",
                "start_tokens",
                "end_tokens",
                "offsets",
            ]:
                if not os.path.exists(
                    os.path.join(cache_dir, f"cached_{split}_{key}.npy")
                ):
                    missing_cache = True
                    break
            # Check for text metadata (parquet)
            if not os.path.exists(
                os.path.join(cache_dir, f"cached_{split}_meta.parquet")
            ):
                missing_cache = True
    else:
        missing_cache = True

    if not missing_cache and load_cached_data:
        print("Loading data from cache...")
        for split in splits:
            data_arrays[split] = {}
            # Load numeric arrays
            data_arrays[split]["input_ids"] = np.load(
                os.path.join(cache_dir, f"cached_{split}_input_ids.npy")
            )
            data_arrays[split]["attention_mask"] = np.load(
                os.path.join(cache_dir, f"cached_{split}_attention_mask.npy")
            )
            data_arrays[split]["start_tokens"] = np.load(
                os.path.join(cache_dir, f"cached_{split}_start_tokens.npy")
            )
            data_arrays[split]["end_tokens"] = np.load(
                os.path.join(cache_dir, f"cached_{split}_end_tokens.npy")
            )
            data_arrays[split]["offsets"] = np.load(
                os.path.join(cache_dir, f"cached_{split}_offsets.npy")
            )

            # Load text/sentiment from Parquet
            df_meta = pd.read_parquet(
                os.path.join(cache_dir, f"cached_{split}_meta.parquet")
            )
            data_arrays[split]["texts"] = df_meta["texts"].values
            data_arrays[split]["sentiments"] = df_meta["sentiments"].values

    else:
        print("Processing data from scratch...")
        # Load raw metadata
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH)
        test_df = pd.read_csv(config.TEST_PATH)

        # Strategic Filtering: Exclude 'neutral' tweets from training and validation
        # The model is only trained to extract positive/negative sentiments.
        train_df = train_df[train_df["sentiment"] != "neutral"].reset_index(drop=True)
        val_df = val_df[val_df["sentiment"] != "neutral"].reset_index(drop=True)

        # Debugging subset
        if config.DEBUG:
            train_df = train_df.head(config.DEBUG_SIZE)
            val_df = val_df.head(config.DEBUG_SIZE)
            test_df = test_df.head(config.DEBUG_SIZE)

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Process each split
        for split, df in zip(splits, [train_df, val_df, test_df]):
            data_arrays[split] = {}
            (
                data_arrays[split]["input_ids"],
                data_arrays[split]["attention_mask"],
                data_arrays[split]["start_tokens"],
                data_arrays[split]["end_tokens"],
                data_arrays[split]["offsets"],
                data_arrays[split]["texts"],
                data_arrays[split]["sentiments"],
            ) = process_data(df, tokenizer, config, mode=split)

            # Save numeric arrays to .npy
            np.save(
                os.path.join(cache_dir, f"cached_{split}_input_ids.npy"),
                data_arrays[split]["input_ids"],
            )
            np.save(
                os.path.join(cache_dir, f"cached_{split}_attention_mask.npy"),
                data_arrays[split]["attention_mask"],
            )
            np.save(
                os.path.join(cache_dir, f"cached_{split}_start_tokens.npy"),
                data_arrays[split]["start_tokens"],
            )
            np.save(
                os.path.join(cache_dir, f"cached_{split}_end_tokens.npy"),
                data_arrays[split]["end_tokens"],
            )
            np.save(
                os.path.join(cache_dir, f"cached_{split}_offsets.npy"),
                data_arrays[split]["offsets"],
            )

            # Save string arrays to Parquet
            df_meta = pd.DataFrame(
                {
                    "texts": data_arrays[split]["texts"],
                    "sentiments": data_arrays[split]["sentiments"],
                }
            )
            df_meta.to_parquet(os.path.join(cache_dir, f"cached_{split}_meta.parquet"))

    # Create Datasets
    datasets = {}
    for split in splits:
        datasets[split] = TweetDataset(
            data_arrays[split]["input_ids"],
            data_arrays[split]["attention_mask"],
            data_arrays[split]["start_tokens"],
            data_arrays[split]["end_tokens"],
            data_arrays[split]["offsets"],
            data_arrays[split]["texts"],
            data_arrays[split]["sentiments"],
        )

    # Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
