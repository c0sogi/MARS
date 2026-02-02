import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything
from library.dataset import PetDataset, get_transforms
from library.models import get_model
from library.engine import train_model


def create_folds(load_cached_data=True):
    """
    Loads metadata, merges train/val splits, creates stratified folds,
    and handles caching of the dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    # 2. Process from scratch
    print("Creating folds from scratch...")

    # Load provided metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)

    # Concatenate to use all available labeled data for CV
    df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # Create Folds
    df["fold"] = -1
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved folds to {cache_path}")

    return df


def run_training(load_cached_data=True):
    """
    Orchestrates the 5-Fold Cross-Validation training for all defined architectures.
    """
    seed_everything(Config.SEED)

    # Prepare Data
    df = create_folds(load_cached_data=load_cached_data)

    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Truncating data to {Config.DEBUG_SUBSET_SIZE} samples."
        )
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    # Iterate over architectures
    for model_name in Config.MODEL_ARCHS:
        print(f"\n{'='*40}")
        print(f"Architecture: {model_name}")
        print(f"{'='*40}")

        # Iterate over folds
        for fold in range(Config.N_FOLDS):
            print(
                f"\n--- Training Fold {fold + 1}/{Config.N_FOLDS} for {model_name} ---"
            )

            # Split Train/Val
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            val_df = df[df["fold"] == fold].reset_index(drop=True)

            # Create Datasets
            train_dataset = PetDataset(
                train_df, transforms=get_transforms(data_type="train"), mode="train"
            )
            val_dataset = PetDataset(
                val_df, transforms=get_transforms(data_type="valid"), mode="train"
            )

            # Create DataLoaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            model = get_model(model_name, pretrained=True, num_classes=1)
            model = model.to(Config.DEVICE)

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Scheduler (Cosine Annealing)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Checkpoint Filename
            checkpoint_name = f"{model_name}_fold_{fold}.pth"

            # Train
            # We set patience equal to EPOCHS to effectively disable early stopping
            # and allow the scheduler to complete its full cycle.
            best_loss = train_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=Config.DEVICE,
                epochs=Config.EPOCHS,
                patience=Config.EPOCHS,
                checkpoint_name=checkpoint_name,
            )

            print(f"Fold {fold} Best Validation Loss: {best_loss}")

            # Cleanup to save memory
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()
