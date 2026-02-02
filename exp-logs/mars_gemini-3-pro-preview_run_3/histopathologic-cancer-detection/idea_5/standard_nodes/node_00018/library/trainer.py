import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.modeling import train_one_epoch, valid_one_epoch, train_model


def get_folds(load_cached_data=True):
    """
    Generates or loads the 5-fold cross-validation split.
    Combines train and validation metadata to use 100% of data.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: DataFrame with a 'fold' column.
    """
    folds_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(folds_path):
        print(f"Loading folds from {folds_path}")
        try:
            return pd.read_parquet(folds_path)
        except Exception as e:
            print(f"Failed to load cached folds: {e}. Regenerating...")

    # 2. IF loading fails OR load_cached_data is False: Compute/process.
    print("Generating new folds...")

    # Ensure metadata directory exists and files are present
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine datasets to use 100% of data for CV
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    df_full["fold"] = -1
    # Stratify by label
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full["label"])):
        df_full.loc[val_idx, "fold"] = fold

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_full.to_parquet(folds_path)

    return df_full


def run_fold(fold_idx, model_name):
    """
    Manages the lifecycle of a single fold training.

    Args:
        fold_idx (int): Index of the fold to train (0 to N_FOLDS-1).
        model_name (str): Name of the model architecture to train.

    Returns:
        tuple: (trained_model, oof_predictions, val_labels)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Preparing data for Fold {fold_idx}...")
    # Get data splits
    df_folds = get_folds(load_cached_data=True)

    # Split data based on fold index
    train_df = df_folds[df_folds["fold"] != fold_idx].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold_idx].reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Get DataLoaders
    loaders = get_dataloaders(
        train_df=train_df, val_df=val_df, batch_size=Config.BATCH_SIZE
    )

    # Run training
    # train_model handles the loop, optimizer, scheduler, early stopping, and saving best model
    model, oof_preds = train_model(
        model_name=model_name,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        fold_idx=fold_idx,
    )

    val_labels = val_df["label"].values

    return model, oof_preds, val_labels
