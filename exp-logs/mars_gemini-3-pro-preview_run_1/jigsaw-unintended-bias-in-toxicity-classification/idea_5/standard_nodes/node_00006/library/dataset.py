import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from library.config import Config
from library.utils import get_logger

logger = get_logger(name="dataset")


class ToxicityDataset(Dataset):
    """
    Dataset class for Toxicity Classification with Semantic Triangulation.
    Returns input_ids, attention_mask, primary targets, auxiliary targets, and bias weights.
    """

    def __init__(
        self,
        texts,
        targets=None,
        identity_targets=None,
        attack_targets=None,
        weights=None,
        tokenizer=None,
        max_len=Config.MAX_LEN,
        is_test=False,
    ):
        self.texts = texts
        self.targets = targets
        self.identity_targets = identity_targets
        self.attack_targets = attack_targets
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize without padding (dynamic padding handled by DataCollator)
        # We truncate to max_len to ensure we don't exceed model limits
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,  # DeBERTa usually doesn't use token_type_ids for single seq
        )

        item = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
            item["identity_targets"] = torch.tensor(
                self.identity_targets[idx], dtype=torch.float
            )
            item["attack_target"] = torch.tensor(
                self.attack_targets[idx], dtype=torch.float
            )
            item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float)

        return item


class CustomCollator:
    """
    Custom data collator that handles dynamic padding for text inputs
    and stacking for auxiliary tensor targets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 1. Separate text inputs (lists) from other fields (Tensors)
        text_inputs = []
        others = {}

        # Initialize lists for other keys
        if len(batch) > 0:
            for k in batch[0].keys():
                if k not in ["input_ids", "attention_mask"]:
                    others[k] = []

        for item in batch:
            # Extract text inputs for padding
            text_inputs.append(
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
            )
            # Collect other fields
            for k in others.keys():
                others[k].append(item[k])

        # 2. Pad text inputs using the tokenizer
        # This returns a BatchEncoding (dict-like) with tensors
        batch_out = self.tokenizer.pad(text_inputs, padding=True, return_tensors="pt")

        # 3. Stack auxiliary tensors
        for k, v_list in others.items():
            # v_list is a list of Tensors. torch.stack creates the batch dimension.
            batch_out[k] = torch.stack(v_list)

        return batch_out


def load_and_preprocess(data_path, mode="train", load_cached_data=True):
    """
    Loads data from CSV, calculates bias weights and targets, and caches the result.

    Args:
        data_path: Path to the metadata CSV file.
        mode: 'train', 'val', or 'test'.
        load_cached_data: Whether to try loading from cache first.

    Returns:
        Dictionary containing processed numpy arrays.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_processed.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading {mode} data from cache: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            return dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing data.")

    # 2. Process from scratch
    logger.info(f"Processing {mode} data from {data_path}...")
    df = pd.read_csv(data_path)

    # Debug mode: subsample
    if Config.DEBUG:
        logger.info(f"DEBUG mode: Subsampling {mode} data.")
        df = df.head(5000)

    # Handle missing text
    df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("")
    texts = df[Config.TEXT_COL].values
    ids = df["id"].values

    result = {"texts": texts, "ids": ids}

    if mode != "test":
        # --- Primary Target ---
        # Use soft targets directly
        targets = df[Config.TARGET_COL].values

        # --- Auxiliary Identity Targets ---
        # Multi-hot vector for identity presence (threshold 0.5)
        # Shape: (N, num_identities)
        identity_cols = Config.IDENTITY_COLS
        # Fill NaNs with 0 for calculation
        df_ident = df[identity_cols].fillna(0.0)
        identity_targets = (df_ident.values >= 0.5).astype(float)

        # --- Auxiliary Attack Target ---
        # Soft target for 'identity_attack' subtype
        if Config.IDENTITY_ATTACK_COL in df.columns:
            attack_targets = df[Config.IDENTITY_ATTACK_COL].fillna(0.0).values
        else:
            attack_targets = np.zeros(len(df))

        # --- Bias-Centric Weights ---
        # Logic: Assign higher weight to "Bias Trap" subgroups.
        # Bias Trap = (Toxic & Identity) OR (Non-Toxic & Identity).
        # This simplifies to: Any example mentioning an identity.
        # We use the same threshold (0.5) to define "mention".
        has_identity = (df_ident >= 0.5).any(axis=1)

        weights = np.ones(len(df), dtype=float)
        weights[has_identity] = Config.BIAS_WEIGHT_MULTIPLIER

        logger.info(f"Bias Weights Stats for {mode}:")
        logger.info(f"  Standard samples (w=1.0): {np.sum(weights == 1.0)}")
        logger.info(
            f"  Bias Trap samples (w={Config.BIAS_WEIGHT_MULTIPLIER}): {np.sum(weights == Config.BIAS_WEIGHT_MULTIPLIER)}"
        )

        result["targets"] = targets
        result["identity_targets"] = identity_targets
        result["attack_targets"] = attack_targets
        result["weights"] = weights

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(cache_file, **result)
    logger.info(f"Saved processed {mode} data to {cache_file}")

    return result


def get_dataloaders(
    train_batch_size=Config.TRAIN_BATCH_SIZE,
    valid_batch_size=Config.VALID_BATCH_SIZE,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Initialize Tokenizer
    logger.info(f"Initializing tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Data
    train_data = load_and_preprocess(Config.TRAIN_META_PATH, "train", load_cached_data)
    val_data = load_and_preprocess(Config.VAL_META_PATH, "val", load_cached_data)
    test_data = load_and_preprocess(Config.TEST_META_PATH, "test", load_cached_data)

    # Create Datasets
    train_dataset = ToxicityDataset(
        texts=train_data["texts"],
        targets=train_data["targets"],
        identity_targets=train_data["identity_targets"],
        attack_targets=train_data["attack_targets"],
        weights=train_data["weights"],
        tokenizer=tokenizer,
        is_test=False,
    )

    val_dataset = ToxicityDataset(
        texts=val_data["texts"],
        targets=val_data["targets"],
        identity_targets=val_data["identity_targets"],
        attack_targets=val_data["attack_targets"],
        weights=val_data["weights"],
        tokenizer=tokenizer,
        is_test=False,
    )

    test_dataset = ToxicityDataset(
        texts=test_data["texts"], tokenizer=tokenizer, is_test=True
    )

    # Use Custom Collator
    collator = CustomCollator(tokenizer=tokenizer)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
