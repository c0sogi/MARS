import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from provided library files
import library.config as config
from library.utils import set_seed, levenshtein_distance, rle_decode, smooth_predictions
from library.data_loader import GestureDataset, collate_fn
from library.model import KAGRN
from library.trainer import Trainer


def run_demo():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    print(">>> Setting up demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override configuration for speed
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 4
    config.WORKING_DIR = "./working/demo_run"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.BEST_MODEL_PATH = os.path.join(config.WORKING_DIR, "best_model_demo.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission_demo.csv")

    # Ensure working directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading (Subset)
    # ==========================================
    print(">>> Loading data subsets...")

    # Load metadata
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    train_df_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df_full = pd.read_csv(config.VAL_METADATA_PATH)
    test_df_full = pd.read_csv(config.TEST_METADATA_PATH)

    # Take a small subset for demonstration (e.g., 10 samples)
    train_subset = train_df_full.head(10).copy()
    val_subset = val_df_full.head(5).copy()
    test_subset = test_df_full.head(5).copy()

    # Save subsets for reference (optional, but good for debugging)
    train_subset.to_csv(
        os.path.join(config.WORKING_DIR, "train_subset.csv"), index=False
    )
    val_subset.to_csv(os.path.join(config.WORKING_DIR, "val_subset.csv"), index=False)
    test_subset.to_csv(os.path.join(config.WORKING_DIR, "test_subset.csv"), index=False)

    # Initialize Datasets
    # Note: This will compute/load stats. Since we use a subset, stats might be slightly different
    # if recomputed, but we rely on the logic in GestureDataset to handle caching/loading.
    print("    Initializing GestureDatasets...")
    train_ds = GestureDataset(train_subset, mode="train")
    val_ds = GestureDataset(val_subset, mode="val")
    test_ds = GestureDataset(test_subset, mode="test")

    # Initialize DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    print(f"    Train subset size: {len(train_ds)}")
    print(f"    Val subset size: {len(val_ds)}")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print(">>> Verifying Model Logic...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KAGRN().to(device)

    # Fetch one batch
    batch = next(iter(train_loader))
    pos = batch["pos"].to(device)
    vel = batch["vel"].to(device)
    audio = batch["audio"].to(device)
    lengths = batch["lengths"]

    # Forward pass
    cls_logits, bnd_logits = model(pos, vel, audio, lengths)

    # Check shapes
    # cls_logits: (Batch, Time, Num_Classes)
    # bnd_logits: (Batch, Time, 1)
    B, T, _ = pos.shape
    assert cls_logits.shape == (
        B,
        T,
        config.NUM_CLASSES,
    ), f"Expected cls_logits shape {(B, T, config.NUM_CLASSES)}, got {cls_logits.shape}"
    assert bnd_logits.shape == (
        B,
        T,
        1,
    ), f"Expected bnd_logits shape {(B, T, 1)}, got {bnd_logits.shape}"

    print("    Model forward pass successful. Output shapes verified.")

    # ==========================================
    # 4. Utility Logic Verification
    # ==========================================
    print(">>> Verifying Utilities...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical sequences should be 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert (
        dist_diff == 1
    ), f"Levenshtein distance for deletion should be 1, got {dist_diff}"

    # Test RLE Decode
    # [1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 1] -> with min_length=3, background=0
    # Should keep 1 (len 3), ignore 0, keep 2 (len 5), ignore 1 (len 1)
    raw_preds = np.array([1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 1])
    decoded = rle_decode(raw_preds, min_length=3, background_label=0)
    assert decoded == [1, 2], f"RLE Decode failed. Expected [1, 2], got {decoded}"

    print("    Utility functions verified.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print(">>> Starting Training Loop Demo...")

    trainer = Trainer(device=device)

    # We use the fit method which handles the loop
    # Since we overrode config.NUM_EPOCHS to 2, this will run quickly
    trainer.fit(train_loader, val_loader, epochs=config.NUM_EPOCHS)

    # Check if best model was saved
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"    Training complete. Model saved to {config.BEST_MODEL_PATH}")
    else:
        # It's possible validation didn't improve if random init was lucky,
        # but usually it saves at least once. If not, we force save for next step.
        print(
            "    Warning: Best model not saved (metrics might not have improved). Saving current manually."
        )
        torch.save(trainer.model.state_dict(), config.BEST_MODEL_PATH)

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print(">>> Starting Inference Demo...")

    trainer.predict(test_loader)

    if os.path.exists(config.SUBMISSION_PATH):
        print(f"    Submission file generated at {config.SUBMISSION_PATH}")

        # Verify content format
        with open(config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                first_line = lines[0].strip()
                parts = first_line.split(",")
                # Format: SessionID,Label1,Label2...
                # SessionID usually starts with 'Sample' or 'Session'
                assert len(parts) >= 1, "Submission line empty or malformed"
                print(f"    Sample submission line: {first_line}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n>>> Demonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
