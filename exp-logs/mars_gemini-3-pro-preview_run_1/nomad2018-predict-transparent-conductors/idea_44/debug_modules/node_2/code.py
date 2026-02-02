import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.features import FeatureExtractor, SelectiveScaler
from library.dataset import MaterialDataset, get_dataloader, sparse_collate_fn
from library.model import AMSP_DS_Net
from library.train import train_one_epoch, evaluate, predict_and_submit


def run_demo():
    print("Initializing AMSP-DS Demo...")

    # 1. Setup & Configuration Override for Speed
    # We override Config attributes to run a small, fast demonstration.
    print("Overriding Config for demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Process only 50 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure clean working directory for demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Feature Extractor Demo
    print("\n--- Feature Extraction Demo ---")
    extractor = FeatureExtractor()

    # Load metadata to get a file path
    train_df = pd.read_csv(Config.TRAIN_CSV).iloc[:5]
    sample_row = train_df.iloc[0]
    file_path = sample_row["file_path"]

    print(f"Parsing geometry file: {file_path}")
    atoms = extractor.parse_xyz(file_path)
    print(f"Parsed atoms: {len(atoms)} atoms")

    # Compute Atomic Features
    atomic_feats = extractor.compute_atomic_features(atoms)
    print(f"Atomic Features shape: {atomic_feats.shape}")
    # Expected: (N_atoms, 17)
    assert (
        atomic_feats.shape[1] == 17
    ), f"Expected 17 atomic features, got {atomic_feats.shape[1]}"

    # Compute Global Features
    global_feats = extractor.compute_global_features(sample_row, atoms)
    print(f"Global Features shape: {global_feats.shape}")
    # Expected: (19,)
    assert global_feats.shape == (
        19,
    ), f"Expected 19 global features, got {global_feats.shape}"

    # 3. Dataset & DataLoader Demo
    print("\n--- Dataset & DataLoader Demo ---")
    # This triggers processing of the debug subset
    print("Initializing Train Dataset (this runs feature extraction)...")
    train_dataset = MaterialDataset(mode="train", load_cached=False)

    print(f"Train dataset size: {len(train_dataset)}")
    assert len(train_dataset) == Config.DEBUG_SIZE

    # Check a single sample
    sample = train_dataset[0]
    print("Sample keys:", sample.keys())
    print("Sample atomic feats shape:", sample["atomic_features"].shape)
    print("Sample global feats shape:", sample["global_features"].shape)
    print("Sample target:", sample["target"])

    # Check DataLoader
    train_loader, scaler = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=False
    )
    print("Train loader created.")

    # Fetch one batch
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Batch atomic_features shape:", batch["atomic_features"].shape)
    print("Batch global_features shape:", batch["global_features"].shape)
    print("Batch batch_indices shape:", batch["batch_indices"].shape)
    print("Batch targets shape:", batch["targets"].shape)

    # Verify batch construction
    # global_features should be (batch_size, 19)
    assert batch["global_features"].shape == (Config.BATCH_SIZE, 19)
    # targets should be (batch_size, 2)
    assert batch["targets"].shape == (Config.BATCH_SIZE, 2)
    # batch_indices length should match atomic_features length
    assert batch["atomic_features"].shape[0] == batch["batch_indices"].shape[0]

    # 4. Model Demo
    print("\n--- Model Architecture Demo ---")
    device = torch.device(
        "cpu"
    )  # Use CPU for demo to avoid overhead/memory issues on small batch
    model = AMSP_DS_Net().to(device)
    print(model)

    # Forward pass with the batch
    af = batch["atomic_features"].to(device)
    gf = batch["global_features"].to(device)
    bi = batch["batch_indices"].to(device)

    output = model(af, gf, bi)
    print("Model output shape:", output.shape)
    assert output.shape == (Config.BATCH_SIZE, 2), "Output shape mismatch"

    # 5. Training Loop Demo
    print("\n--- Training Loop Demo ---")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run 1 epoch of training
    print("Running training for one epoch...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.6f}")

    # Run evaluation
    print("Initializing Val Dataset...")
    val_loader = get_dataloader(
        "val", batch_size=Config.BATCH_SIZE, shuffle=False, scaler=scaler
    )

    print("Running evaluation...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print("Validation Metrics:", val_metrics)
    assert "val_loss" in val_metrics
    assert "rmsle_mean" in val_metrics

    # 6. Prediction & Submission Demo
    print("\n--- Prediction & Submission Demo ---")
    # We mock the predict_and_submit function call by calling it directly
    # It will load 'test' data. Since DEBUG is True, it loads subset of test.csv

    # Ensure test metadata exists (it should from the prompt description)
    if os.path.exists(Config.TEST_CSV):
        print("Test metadata found. Running prediction pipeline...")
        # We need to save the model first because predict_and_submit loads it
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pt")
        torch.save(model.state_dict(), model_path)

        # Call the function (it reloads the model internally)
        # Note: predict_and_submit expects the scaler to be passed
        predict_and_submit(model, scaler, device)

        if os.path.exists(Config.SUBMISSION_FILE):
            print(f"Submission file successfully generated at {Config.SUBMISSION_FILE}")
            df_sub = pd.read_csv(Config.SUBMISSION_FILE)
            print("Submission head:")
            print(df_sub.head())
            assert df_sub.shape[1] == 3, "Submission should have 3 columns"
            assert "id" in df_sub.columns
            assert "formation_energy_ev_natom" in df_sub.columns
            assert "bandgap_energy_ev" in df_sub.columns
        else:
            raise FileNotFoundError("Submission file was not created.")
    else:
        print("Test metadata not found, skipping prediction demo.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
