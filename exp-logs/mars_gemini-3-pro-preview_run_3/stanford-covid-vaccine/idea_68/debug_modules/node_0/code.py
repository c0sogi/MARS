import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
import library.config
from library.config import Config
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_model
from library.utils import seed_everything, mcrmse_loss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing RNA Degradation Prediction Demo...")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Define a lightweight configuration for the demonstration
    class DemoConfig(Config):
        # Paths
        CACHE_DIR = "./working/demo_execution/cache/"
        SUBMISSION_DIR = "./working/demo_execution/"

        # Reduced Model Hyperparameters for Speed
        STEM_FILTERS = 16
        HIDDEN_DIM = 32
        TOTAL_HIDDEN = 64  # 2 * HIDDEN_DIM
        LAYERS = 2

        # Training Settings
        EPOCHS = 2
        BATCH_SIZE = 16
        NUM_WORKERS = 0  # Avoid multiprocessing overhead
        PATIENCE = 2
        SEED = 42

    # Monkeypatch the global Config in library.config so internal functions
    # (like process_dataframe) use our demo cache directory.
    library.config.Config.CACHE_DIR = DemoConfig.CACHE_DIR

    # Clean up any previous demo run
    if os.path.exists("./working/demo_execution"):
        shutil.rmtree("./working/demo_execution")
    os.makedirs(DemoConfig.CACHE_DIR, exist_ok=True)
    os.makedirs(DemoConfig.SUBMISSION_DIR, exist_ok=True)

    # Set random seed
    seed_everything(DemoConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n--- Data Loading ---")
    # get_dataloaders processes metadata parquet files and caches them as .npz
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=DemoConfig.BATCH_SIZE,
        num_workers=DemoConfig.NUM_WORKERS,
        load_cached_data=True,
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    features = batch["features"]
    pair_indices = batch["pair_indices"]
    targets = batch["targets"]

    print(f"Batch Features Shape: {features.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    # Assertions to ensure data integrity
    # Features: (Batch, Seq_Len=107, Channels=14)
    assert features.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
        DemoConfig.INPUT_CHANNELS,
    ), f"Feature shape mismatch: {features.shape}"
    # Targets: (Batch, Seq_Len=107, Targets=5)
    assert targets.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
        5,
    ), f"Target shape mismatch: {targets.shape}"
    # Pair Indices: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
    ), f"Pair indices shape mismatch: {pair_indices.shape}"

    print("Data loading verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n--- Model Initialization & Forward Pass ---")
    model = RNAModel(DemoConfig).to(device)

    # Move batch to device
    features = features.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = batch["pair_masks"].to(device)
    targets = targets.to(device)

    # Execute Forward Pass
    preds = model(features, pair_indices, pair_masks)
    print(f"Predictions Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        DemoConfig.BATCH_SIZE,
        DemoConfig.SEQ_LEN,
        5,
    ), "Prediction output shape is incorrect."
    assert torch.isfinite(preds).all(), "Model output contains NaN or Inf values."

    # Calculate Loss (MCRMSE)
    # Note: We slice to SEQ_SCORED (68) as per competition rules for the loss
    preds_scored = preds[:, : DemoConfig.SEQ_SCORED, :]
    targets_scored = targets[:, : DemoConfig.SEQ_SCORED, :]

    loss = mcrmse_loss(preds_scored, targets_scored)
    print(f"Initial Batch Loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss cannot be negative."

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n--- Training Loop Execution ---")
    save_path = os.path.join(DemoConfig.CACHE_DIR, "best_model.pth")

    # train_model handles the training loop, validation, and saving
    trained_model = train_model(
        config=DemoConfig, load_cached_data=True, save_path=save_path
    )

    assert os.path.exists(save_path), "Best model checkpoint was not created."
    print("Training loop completed successfully.")

    # ---------------------------------------------------------
    # 5. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n--- Inference & Submission ---")
    trained_model.eval()

    all_preds = []
    all_ids = []

    # Run Inference on Test Set
    with torch.no_grad():
        for batch in test_loader:
            f = batch["features"].to(device)
            pi = batch["pair_indices"].to(device)
            pm = batch["pair_masks"].to(device)
            ids = batch["id"]

            p = trained_model(f, pi, pm)

            all_preds.append(p.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # Shape: (N_test, 107, 5)

    # Verify Test Dimensions (Test set has 240 samples)
    assert len(all_ids) == 240, f"Expected 240 test samples, got {len(all_ids)}"
    assert all_preds.shape == (
        240,
        107,
        5,
    ), f"Expected (240, 107, 5), got {all_preds.shape}"

    # Format Submission CSV
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        preds_i = all_preds[i]  # (107, 5)
        for pos in range(DemoConfig.SEQ_LEN):
            id_seqpos = f"{sample_id}_{pos}"
            row = [id_seqpos] + preds_i[pos].tolist()
            submission_rows.append(row)

    columns = ["id_seqpos"] + target_cols
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    # Save Submission
    sub_path = os.path.join(DemoConfig.SUBMISSION_DIR, "submission_demo.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(f"Submission Shape: {sub_df.shape}")

    # Final Verification
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    print("\nDemonstration Complete.")


if __name__ == "__main__":
    run_demo()
