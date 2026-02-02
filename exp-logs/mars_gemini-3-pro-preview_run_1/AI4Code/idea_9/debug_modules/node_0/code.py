import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ==========================================
# 1. Patch TQDM to be silent (Requirement: No progress bars)
# ==========================================
# This must be done before importing library modules that use tqdm
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# ==========================================
# 2. Imports from Library
# ==========================================
from library.config import Config
from library.preprocess import preprocess_data
from library.dataset import CachedNotebookDataset, custom_collate_fn
from library.model import DualContextAnchorNetwork
from library.engine import train_model
from library.inference import predict_and_rank

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Setup & Reproducibility ---
    print(">>> Setting up environment and configuration...")
    torch.manual_seed(42)
    np.random.seed(42)

    # Override Config for a fast demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.NUM_EPOCHS = 1
    Config.PATIENCE = 1

    # Update dependent paths manually as they were initialized at class level
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Re-run setup to create the new working directory
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")

    # --- Step 1: Preprocessing ---
    print("\n>>> Step 1: Running Preprocessing (Debug Mode)...")
    # debug=True limits processing to the first 100 notebooks for speed
    preprocess_data(debug=True, load_cached_data=False)

    # Validation: Check if feature files exist
    assert os.path.exists(Config.TRAIN_FEATURES_PATH), "Train features parquet missing!"
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val features parquet missing!"
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test features parquet missing!"
    print("Preprocessing verification successful: Parquet files created.")

    # --- Step 2: Dataset & DataLoader ---
    print("\n>>> Step 2: Verifying Dataset and DataLoader...")
    # Load the debug train dataset
    train_ds = CachedNotebookDataset(split="train", debug=True)

    # Validation: Dataset should not be empty
    assert len(train_ds) > 0, "Training dataset is empty."
    print(f"Loaded {len(train_ds)} notebooks in training dataset.")

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=4,  # Small batch for demo
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        collate_fn=custom_collate_fn,
    )

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))

    # Validation: Check key keys in batch
    required_keys = [
        "code_embeddings",
        "markdown_embeddings",
        "labels",
        "code_padding_mask",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key {key} in batch."

    # Validation: Check Embedding Dimensions (MPNet is 768)
    assert (
        batch["code_embeddings"].shape[-1] == 768
    ), "Incorrect code embedding dimension."
    assert (
        batch["markdown_embeddings"].shape[-1] == 768
    ), "Incorrect markdown embedding dimension."
    print("DataLoader verification successful: Batch shapes correct.")

    # --- Step 3: Model Initialization & Forward Pass ---
    print("\n>>> Step 3: Verifying Model Architecture...")
    device = Config.DEVICE
    model = DualContextAnchorNetwork().to(device)

    # Move batch to device
    code_emb = batch["code_embeddings"].to(device)
    code_lens = batch["code_lens"].to(device)
    code_mask = batch["code_padding_mask"].to(device)
    md_emb = batch["markdown_embeddings"].to(device)
    md_lens = batch["md_lens"].to(device)
    md_mask = batch["md_padding_mask"].to(device)

    # Forward pass
    logits = model(code_emb, code_lens, code_mask, md_emb, md_lens, md_mask)

    # Validation: Output shape should be (Batch, Max_MD, Max_Code + 1)
    # Max_Code + 1 accounts for the EOS token appended in the model
    batch_size = code_emb.size(0)
    max_md = md_emb.size(1)
    max_code = code_emb.size(1)

    assert logits.shape == (
        batch_size,
        max_md,
        max_code + 1,
    ), f"Expected shape {(batch_size, max_md, max_code + 1)}, got {logits.shape}"

    print("Model verification successful: Forward pass output shape correct.")

    # --- Step 4: Training Loop ---
    print("\n>>> Step 4: Demonstrating Training Loop (1 Epoch)...")

    # Setup Validation Loader
    val_ds = CachedNotebookDataset(split="validation", debug=True)
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Validation: Check if model checkpoint was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved."
    print("Training verification successful: Model trained and saved.")

    # --- Step 5: Inference ---
    print("\n>>> Step 5: Running Inference and Generating Submission...")

    # Run inference (debug=True uses first 100 test notebooks)
    predict_and_rank(debug=True)

    # Validation: Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check format of cell_order (space delimited string)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order is not a string."
    assert len(sample_order.split()) > 0, "cell_order string is empty."

    print(f"Inference verification successful: Generated {len(df_sub)} predictions.")
    print(f"Sample Prediction: {df_sub.iloc[0].to_dict()}")

    print("\n>>> All demonstrations completed successfully.")
