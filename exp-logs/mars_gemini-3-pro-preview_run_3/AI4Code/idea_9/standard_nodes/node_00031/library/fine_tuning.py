import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from library.config import Config
from library.data_loader import create_relaxed_pairs, FineTuningDataset


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_semantic_model(load_cached_data: bool = True):
    """
    Fine-tunes the SentenceTransformer backbone using the Relaxed Proximity strategy.

    Args:
        load_cached_data: If True, attempts to load processed data/model from disk.
    """
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if model already exists to avoid retraining
    if load_cached_data and os.path.exists(Config.MODEL_OUTPUT_PATH):
        print(
            f"Fine-tuned model already exists at {Config.MODEL_OUTPUT_PATH}. Skipping training."
        )
        return

    print("Initializing Semantic Model Training...")

    # 2. Prepare Data
    # Load training metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Generate pairs (handles caching internally)
    pairs_df = create_relaxed_pairs(df_train, load_cached_data=load_cached_data)

    print(f"Training on {len(pairs_df)} pairs.")

    # Create Dataset and DataLoader
    train_dataset = FineTuningDataset(pairs_df)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # 3. Initialize Model
    print(f"Loading backbone model: {Config.MODEL_NAME}")
    model = SentenceTransformer(Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.max_seq_length = Config.MAX_LENGTH

    # 4. Define Loss
    # MultipleNegativesRankingLoss is effective for (query, positive) pairs
    # It uses other samples in the batch as negatives.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. Train
    # Calculate warmup steps
    # If Config.WARMUP_STEPS is absolute, use it.
    # Otherwise, typically 10% of train data is good, but we stick to Config.

    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=Config.EPOCHS,
        warmup_steps=Config.WARMUP_STEPS,
        optimizer_params={"lr": Config.LEARNING_RATE},
        weight_decay=Config.WEIGHT_DECAY,
        output_path=Config.MODEL_OUTPUT_PATH,
        show_progress_bar=True,
        use_amp=True,  # Use Automatic Mixed Precision for speed on A100
    )

    print(f"Model saved to {Config.MODEL_OUTPUT_PATH}")
