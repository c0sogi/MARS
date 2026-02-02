import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import (
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    DEVICE,
)
from library.data_utils import process_data, CrystalDataset, collate_fn, StandardScaler
from library.model import CADSTFModel
from library.train import run_training, generate_submission, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing demonstration...")

    # 1. Set Seed
    set_seed(SEED)
    print(f"Random seed set to {SEED}")

    # Configuration for quick demonstration
    DEMO_MAX_SAMPLES = 50  # Process only 50 samples to save time
    DEMO_EPOCHS = 2  # Train for only 2 epochs
    DEMO_BATCH_SIZE = 4

    # ---------------------------------------------------------
    # 2. Data Processing Demonstration
    # ---------------------------------------------------------
    print("\n[Step 1] Processing Data...")

    # Process training data
    # We force reprocessing to ensure we test the parsing logic
    # In a real run, load_cached_data=True is preferred
    train_data = process_data(
        TRAIN_METADATA_PATH,
        TRAIN_CACHE_PATH,
        load_cached_data=False,
        max_samples=DEMO_MAX_SAMPLES,
    )

    # Validations for Data Processing
    assert "global_features" in train_data
    assert "atomic_features_flat" in train_data
    assert "atom_counts" in train_data
    assert "targets" in train_data
    assert "ids" in train_data

    num_samples = len(train_data["ids"])
    print(f"  Processed {num_samples} training samples.")
    assert (
        num_samples == DEMO_MAX_SAMPLES
    ), f"Expected {DEMO_MAX_SAMPLES} samples, got {num_samples}"

    # Check shapes
    # Global features should be (N, 11) based on config
    assert train_data["global_features"].shape == (num_samples, 11)
    # Targets should be (N, 2)
    assert train_data["targets"].shape == (num_samples, 2)

    # Process Validation Data (Cached if available, but we limit samples)
    val_data = process_data(
        VAL_METADATA_PATH,
        VAL_CACHE_PATH,
        load_cached_data=False,
        max_samples=DEMO_MAX_SAMPLES,
    )
    print(f"  Processed {len(val_data['ids'])} validation samples.")

    # ---------------------------------------------------------
    # 3. Scaler and Dataset Demonstration
    # ---------------------------------------------------------
    print("\n[Step 2] Dataset and Scaler...")

    # Initialize and fit scaler
    scaler = StandardScaler()
    scaler.fit(
        torch.tensor(train_data["global_features"], dtype=torch.float32),
        torch.tensor(train_data["atomic_features_flat"], dtype=torch.float32),
    )
    print("  Scaler fitted.")

    # Instantiate Datasets
    train_dataset = CrystalDataset(train_data, scaler=scaler)
    val_dataset = CrystalDataset(val_data, scaler=scaler)

    # Validate Dataset Item
    sample_item = train_dataset[0]
    print("  Sample item keys:", sample_item.keys())
    assert "global_features" in sample_item
    assert "atomic_features" in sample_item
    assert "targets" in sample_item
    assert "id" in sample_item

    # Check if atomic features are scaled (not strictly zero mean due to one-hot, but check type)
    assert isinstance(sample_item["atomic_features"], torch.Tensor)
    assert sample_item["atomic_features"].ndim == 2  # (Atoms, Features)

    # ---------------------------------------------------------
    # 4. DataLoader and Collate Function
    # ---------------------------------------------------------
    print("\n[Step 3] DataLoader and Batching...")

    train_loader = DataLoader(
        train_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print("  Batch keys:", batch.keys())

    # Validate Batch Shapes
    # Global: (Batch, 11)
    assert batch["global_features"].shape == (DEMO_BATCH_SIZE, 11)
    # Atomic: (Batch, MaxAtomsInBatch, FeatureDim)
    assert batch["atomic_features"].ndim == 3
    assert batch["atomic_features"].shape[0] == DEMO_BATCH_SIZE
    # Mask: (Batch, MaxAtomsInBatch)
    assert batch["mask"].ndim == 2
    assert batch["mask"].shape == batch["atomic_features"].shape[:2]
    # Targets: (Batch, 2)
    assert batch["targets"].shape == (DEMO_BATCH_SIZE, 2)

    print("  Batch shapes validated.")

    # ---------------------------------------------------------
    # 5. Model Instantiation and Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 4] Model Architecture...")

    model = CADSTFModel().to(DEVICE)
    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model instantiated with {param_count:,} parameters.")

    # Move batch to device
    g_feats = batch["global_features"].to(DEVICE)
    a_feats = batch["atomic_features"].to(DEVICE)
    mask = batch["mask"].to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(g_feats, a_feats, mask)

    print(f"  Output shape: {outputs.shape}")
    assert outputs.shape == (DEMO_BATCH_SIZE, 2)
    print("  Forward pass successful.")

    # ---------------------------------------------------------
    # 6. Full Training Loop Execution
    # ---------------------------------------------------------
    print("\n[Step 5] Running Training Loop (Dry Run)...")

    # We use the library function run_training which encapsulates the loop
    # We pass max_samples and num_epochs to keep it fast
    trained_scaler = run_training(max_samples=DEMO_MAX_SAMPLES, num_epochs=DEMO_EPOCHS)

    # Check if model file was created
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"  Model saved successfully at {MODEL_SAVE_PATH}")
    else:
        # It might not save if validation doesn't improve, but in epoch 1 it usually sets best
        # For this demo, we assume it might run. If not, we save manually to test submission.
        print(
            "  Model checkpointer didn't trigger (expected if loss didn't improve), saving manually for test."
        )
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 6] Generating Submission...")

    # We use the scaler returned by run_training (or our local one if that failed)
    final_scaler = trained_scaler if trained_scaler is not None else scaler

    # Run the submission generation function
    # We limit max_samples for test data processing as well to be quick
    generate_submission(scaler=final_scaler, max_samples=DEMO_MAX_SAMPLES)

    # Validate submission file
    if os.path.exists(SUBMISSION_PATH):
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"  Submission loaded. Shape: {df_sub.shape}")

        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert list(df_sub.columns) == expected_cols

        # Check for NaNs
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"

        # Check values are numeric
        assert pd.api.types.is_numeric_dtype(df_sub["formation_energy_ev_natom"])
        assert pd.api.types.is_numeric_dtype(df_sub["bandgap_energy_ev"])

        print("  Submission format valid.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
