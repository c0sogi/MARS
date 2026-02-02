import os
import sys
import torch
import pandas as pd
import numpy as np
import random
import shutil

# Import provided library modules
from library.config import Config
from library.preprocess import Preprocessor
from library.dataset import CachedNotebookDataset, collate_fn
from library.model import DC_AN
from library.train import train_one_epoch, validate
from library.inference import run_inference


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment for Demo Run...")

    # Override Config attributes for a fast, minimal demonstration
    Config.WORKING_DIR = "./working/demo_run"
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Only process 50 notebooks per split
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo
    Config.SEED = 42

    # Update paths in Config based on the new working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG} (Samples: {Config.DEBUG_SAMPLES})")

    # ---------------------------------------------------------
    # 2. Preprocessing (Feature Generation)
    # ---------------------------------------------------------
    print("\n[2] Running Preprocessing...")

    # Clean up any existing demo files to ensure fresh generation
    for path in [
        Config.TRAIN_FEATURES_PATH,
        Config.VAL_FEATURES_PATH,
        Config.TEST_FEATURES_PATH,
    ]:
        if os.path.exists(path):
            os.remove(path)

    # Instantiate Preprocessor
    # It reads configuration from the Config class we just modified
    preprocessor = Preprocessor()

    # Generate embeddings (Train, Val, Test)
    # load_cached_data=False forces regeneration
    preprocessor.generate_embeddings(load_cached_data=False)

    # Validation: Check if files were created
    assert os.path.exists(Config.TRAIN_FEATURES_PATH), "Train parquet file missing!"
    assert os.path.exists(Config.VAL_FEATURES_PATH), "Val parquet file missing!"
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test parquet file missing!"

    # Validation: Check content size
    df_train_check = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
    print(f"    Generated {len(df_train_check)} training samples.")
    assert (
        len(df_train_check) <= Config.DEBUG_SAMPLES
    ), "Preprocessing did not respect DEBUG_SAMPLES limit."
    assert "code_embeddings" in df_train_check.columns
    assert "markdown_embeddings" in df_train_check.columns

    # ---------------------------------------------------------
    # 3. Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n[3] Testing Dataset and DataLoader...")

    train_ds = CachedNotebookDataset(Config.TRAIN_FEATURES_PATH, Config)
    val_ds = CachedNotebookDataset(Config.VAL_FEATURES_PATH, Config)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Unpack
    code_emb = batch["code_emb"]
    md_emb = batch["md_emb"]
    md_mask = batch["md_mask"]
    labels = batch["labels"]

    print(f"    Batch Size: {len(batch['ids'])}")
    print(f"    Code Emb Shape: {code_emb.shape}")
    print(f"    MD Emb Shape: {md_emb.shape}")

    # Assertions
    assert code_emb.dim() == 3, "Code embeddings should be 3D [B, L, D]"
    assert code_emb.shape[2] == Config.INPUT_DIM, f"Expected dim {Config.INPUT_DIM}"

    # Handle case where batch might have 0 markdown cells (unlikely but possible)
    if md_emb.shape[1] > 0:
        assert md_emb.dim() == 3
        assert md_mask.dim() == 2
        assert labels.dim() == 2

    # ---------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Initializing Model and Running Forward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    model = DC_AN(Config).to(device)

    # Move batch to device
    b_code_emb = code_emb.to(device)
    b_code_lens = batch["code_lens"].to(device)
    b_md_emb = md_emb.to(device)
    b_md_mask = md_mask.to(device)

    # Forward pass
    logits = model(b_code_emb, b_code_lens, b_md_emb, b_md_mask)

    # Expected Output: [Batch, Max_MD_Len, Max_Code_Len + 1]
    # Note: L+1 because of the EOS token appended to code sequence
    L = b_code_emb.size(1)
    M = b_md_emb.size(1)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (len(batch["ids"]), M, L + 1), "Incorrect logits shape"

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Run one epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

    print(f"    Epoch Completed. Average Loss: {avg_loss:.4f}")
    assert isinstance(avg_loss, float), "Loss should be a float"
    assert avg_loss >= 0, "Loss should be non-negative"

    # ---------------------------------------------------------
    # 6. Validation Demonstration
    # ---------------------------------------------------------
    print("\n[6] Running Validation...")

    # Load metadata for ground truth lookup
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    val_score = validate(model, val_loader, df_val_meta, device)
    print(f"    Validation Kendall Tau Score: {val_score:.4f}")

    # Save the model weights for the inference step
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model failed to save"

    # ---------------------------------------------------------
    # 7. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[7] Running Inference Pipeline...")

    # run_inference handles loading the test set, loading the model, and generating the CSV
    run_inference(
        config=Config,
        load_cached_data=True,  # Use the test features generated in step 2
        output_path=Config.SUBMISSION_PATH,
        model_path=Config.MODEL_SAVE_PATH,
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(df_sub)} rows.")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "cell_order" in df_sub.columns, "Submission missing 'cell_order' column"

    # Check content of first prediction
    if len(df_sub) > 0:
        first_order = df_sub.iloc[0]["cell_order"]
        assert isinstance(first_order, str), "cell_order must be a string"
        assert len(first_order.split()) > 0, "cell_order should not be empty"

    print("\n=== Success! All components verified. ===")


if __name__ == "__main__":
    main()
