import os
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from library.config import Config
from library.neural_net import (
    CharTokenizer,
    TargetBPETokenizer,
    ResidualDataset,
    ResidualGenerator,
)
from library.hfbb_engine import HFBB
from library.utils import is_semiotic

# Alias the provided dataset class as requested
NormalizationDataset = ResidualDataset


class Tokenizer:
    """
    Wrapper class for managing character and BPE vocabularies.
    """

    def __init__(self):
        self.char = CharTokenizer()
        self.bpe = TargetBPETokenizer()

    def fit(self, df: pd.DataFrame):
        """
        Fits the tokenizers on the provided dataframe.
        Args:
            df: Dataframe containing 'before', 'prev', 'next', and 'after' columns.
        """
        print("Fitting tokenizers...")
        # Fit CharTokenizer on all input text fields
        sources = (
            df["before"].astype(str).tolist()
            + df["prev"].astype(str).tolist()
            + df["next"].astype(str).tolist()
        )
        self.char.fit(sources)

        # Train BPE Tokenizer on targets
        if "after" in df.columns:
            targets = df["after"].astype(str).tolist()
            self.bpe.train(targets)
        else:
            print(
                "Warning: 'after' column missing, skipping BPE training (assuming inference mode or pre-trained)."
            )


class CollateFn:
    """
    Collate function for padding batches.
    """

    def __init__(self, char_pad_id: int, bpe_pad_id: int):
        self.char_pad_id = char_pad_id
        self.bpe_pad_id = bpe_pad_id

    def __call__(self, batch):
        # Check if batch contains tuples (src, tgt) or just src
        if isinstance(batch[0], tuple):
            src_batch = [item[0] for item in batch]
            tgt_batch = [item[1] for item in batch]

            src_padded = pad_sequence(
                src_batch, batch_first=True, padding_value=self.char_pad_id
            )
            tgt_padded = pad_sequence(
                tgt_batch, batch_first=True, padding_value=self.bpe_pad_id
            )
            return src_padded, tgt_padded
        else:
            # Inference mode (only source)
            src_batch = batch
            src_padded = pad_sequence(
                src_batch, batch_first=True, padding_value=self.char_pad_id
            )
            return src_padded


def get_enriched_residuals(
    mode: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads residuals and enriches them with correct context (prev/next) from the full dataset.
    This fixes the issue where ResidualDataset computes context without respecting sentence boundaries.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"enriched_residual_{mode}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading enriched {mode} residuals from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Generating enriched {mode} residuals...")

    # 1. Get raw residuals (cache misses)
    if mode == "train":
        residuals = ResidualGenerator.get_train_residuals(
            load_cached_data=load_cached_data
        )
        source_path = Config.TRAIN_FILE
    else:
        residuals = ResidualGenerator.get_val_residuals(
            load_cached_data=load_cached_data
        )
        source_path = Config.VAL_FILE

    if len(residuals) == 0:
        return residuals

    # 2. Load full source data to compute correct context
    full_df = pd.read_csv(source_path)

    # 3. Compute context respecting sentence boundaries
    full_df["before"] = full_df["before"].astype(str)
    full_df["prev"] = full_df["before"].shift(1).fillna("<START>")
    full_df["next"] = full_df["before"].shift(-1).fillna("<END>")

    # Mask boundaries
    if "sentence_id" in full_df.columns:
        is_start = full_df["sentence_id"] != full_df["sentence_id"].shift(1)
        full_df.loc[is_start, "prev"] = "<START>"
        is_end = full_df["sentence_id"] != full_df["sentence_id"].shift(-1)
        full_df.loc[is_end, "next"] = "<END>"

    # 4. Merge context into residuals
    # Ensure dtypes match for merge keys
    residuals["sentence_id"] = residuals["sentence_id"].astype(
        full_df["sentence_id"].dtype
    )
    residuals["token_id"] = residuals["token_id"].astype(full_df["token_id"].dtype)

    context_df = full_df[["sentence_id", "token_id", "prev", "next"]]

    # Drop existing prev/next in residuals if they exist (likely incorrect or missing)
    if "prev" in residuals.columns:
        residuals = residuals.drop(columns=["prev"])
    if "next" in residuals.columns:
        residuals = residuals.drop(columns=["next"])

    enriched = pd.merge(
        residuals, context_df, on=["sentence_id", "token_id"], how="left"
    )

    # 5. Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    enriched.to_parquet(cache_file, index=False)

    return enriched


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    debug: bool = Config.DEBUG,
):
    """
    Prepares data and returns dataloaders for training and validation.
    """
    # Load Data
    df_train = get_enriched_residuals(mode="train")
    df_val = get_enriched_residuals(mode="val")

    if debug:
        print(f"DEBUG: Truncating datasets to {Config.DEBUG_SIZE}")
        df_train = df_train.iloc[: Config.DEBUG_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SIZE]

    # Initialize and Fit Tokenizer
    tokenizer = Tokenizer()
    # Fit on training data
    tokenizer.fit(df_train)

    # Create Datasets
    # Note: We pass tokenizer.char and tokenizer.bpe to satisfy ResidualDataset signature
    train_dataset = NormalizationDataset(
        df_train, tokenizer.char, tokenizer.bpe, train_mode=True
    )
    val_dataset = NormalizationDataset(
        df_val, tokenizer.char, tokenizer.bpe, train_mode=True
    )

    # Create Collate Function
    collate_fn = CollateFn(
        char_pad_id=tokenizer.char.pad_token_id, bpe_pad_id=tokenizer.bpe.pad_id
    )

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer


def prepare_test_candidates(test_file: str = Config.TEST_FILE) -> pd.DataFrame:
    """
    Generates the candidate dataset for the test set (Tier 2 inputs).
    Identifies tokens where Tier 1 (HFBB) fails AND the token is semiotic.
    """
    print("Preparing test candidates...")
    df_test = pd.read_csv(test_file)

    # Load trained HFBB
    hfbb = HFBB()
    hfbb.fit(load_cached_data=True)  # Assumes HFBB is already trained/cached

    # Run HFBB prediction (Tier 1)
    # _vectorized_predict adds prev/next columns internally to a copy
    preds = ResidualGenerator._vectorized_predict(hfbb, df_test)

    # Identify candidates: HFBB returns NaN (miss) AND token is semiotic
    is_miss = preds.isna()
    is_semiotic_mask = df_test["before"].astype(str).apply(is_semiotic)

    candidates_mask = is_miss & is_semiotic_mask

    # We need to reconstruct the dataframe with context for the candidates
    # Since _vectorized_predict does not return the context-enriched df, we must do it manually
    # similar to get_enriched_residuals, but for test set.

    # 1. Compute context
    df_test["before"] = df_test["before"].astype(str)
    df_test["prev"] = df_test["before"].shift(1).fillna("<START>")
    df_test["next"] = df_test["before"].shift(-1).fillna("<END>")

    if "sentence_id" in df_test.columns:
        is_start = df_test["sentence_id"] != df_test["sentence_id"].shift(1)
        df_test.loc[is_start, "prev"] = "<START>"
        is_end = df_test["sentence_id"] != df_test["sentence_id"].shift(-1)
        df_test.loc[is_end, "next"] = "<END>"

    # 2. Filter
    candidates = df_test[candidates_mask].copy()

    print(
        f"Found {len(candidates)} candidates for Tier 2 inference out of {len(df_test)} total tokens."
    )
    return candidates
