import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.preprocessing import get_preprocessed_data
from library.dataset import ManufacturingDataset
from library.model import TreeFunnelEnsemble
from library.engine import train_model, generate_submission, set_seed


def main():
    print("Starting Demo Execution...")

    # 1. Setup & Configuration Overrides for Speed
    # We modify the Config class attributes directly to control the execution flow
    # without modifying the source file.
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 256  # Smaller batch size for the small subset
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure demo directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set global seed
    set_seed(Config.SEED)

    # 2. Data Loading & Preprocessing
    print("Loading and preprocessing data...")
    # This will load from cache if available or process from scratch
    train_data, val_data, test_data, vocab_sizes = get_preprocessed_data(
        load_from_cache=True
    )

    # 3. Create Subsets for Speed (Demo Requirement)
    print("Creating data subsets for rapid demonstration...")
    SUBSET_SIZE = 2000

    # Helper to slice dictionary of arrays
    def slice_data(data_dict, size, is_test=False):
        new_dict = {}
        current_len = len(data_dict["cat"])
        indices = np.arange(min(size, current_len))

        new_dict["cat"] = data_dict["cat"][indices]
        new_dict["cont"] = data_dict["cont"][indices]

        if is_test:
            new_dict["ids"] = data_dict["ids"][indices]
        else:
            new_dict["target"] = data_dict["target"][indices]
        return new_dict

    train_subset = slice_data(train_data, SUBSET_SIZE)
    val_subset = slice_data(val_data, SUBSET_SIZE)
    test_subset = slice_data(test_data, SUBSET_SIZE, is_test=True)

    print(f"Train subset shape: {train_subset['cat'].shape}")
    print(f"Val subset shape:   {val_subset['cat'].shape}")
    print(f"Test subset shape:  {test_subset['cat'].shape}")

    # 4. Create Datasets and Dataloaders
    print("Initializing Datasets and Dataloaders...")

    train_dataset = ManufacturingDataset(
        train_subset["cat"], train_subset["cont"], train_subset["target"]
    )
    val_dataset = ManufacturingDataset(
        val_subset["cat"], val_subset["cont"], val_subset["target"]
    )
    test_dataset = ManufacturingDataset(
        test_subset["cat"], test_subset["cont"], targets=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 5. Model Verification (Architecture Check)
    print("Verifying Model Architecture...")
    cont_dim = train_subset["cont"].shape[1]
    model = TreeFunnelEnsemble(vocab_sizes, cont_dim)

    # Create a dummy batch to check forward pass
    dummy_cat = torch.tensor(train_subset["cat"][:2], dtype=torch.long)
    dummy_cont = torch.tensor(train_subset["cont"][:2], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_cat, dummy_cont)

    # Check output structure
    assert isinstance(
        outputs, list
    ), "Model output should be a list (one for each head)"
    assert (
        len(outputs) == Config.NUM_HEADS
    ), f"Model should have {Config.NUM_HEADS} heads"
    assert outputs[0].shape == (
        2,
        1,
    ), f"Output shape mismatch. Expected (2, 1), got {outputs[0].shape}"
    print("Model architecture verification passed.")

    # 6. Training Loop
    print("Starting Training...")
    # train_model handles the loop, validation, and saving best model
    best_auc = train_model(train_loader, val_loader, vocab_sizes, cont_dim)

    # Verify training result
    assert os.path.exists(Config.MODEL_PATH), "Best model file was not saved."
    assert 0.0 <= best_auc <= 1.0, f"AUC score {best_auc} is out of bounds."
    print(f"Training finished successfully with Best AUC: {best_auc:.4f}")

    # 7. Inference and Submission
    print("Generating Submission...")
    test_ids = test_subset["ids"]
    generate_submission(test_loader, test_ids, vocab_sizes, cont_dim)

    # 8. Final Verification of Submission File
    print("Verifying Submission File...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    expected_rows = len(test_subset["ids"])
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing."

    # Check ID correctness
    assert df_sub["id"].iloc[0] == test_ids[0], "ID mismatch in submission."

    # Check probability range
    preds = df_sub["target"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]."

    print("All checks passed. Demo execution complete.")


if __name__ == "__main__":
    main()
