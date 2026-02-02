import os
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.data import GeometryParser, MaterialDataset, collate_fn, get_dataloaders
from library.model import CR_WDS
from library.train import Trainer, set_seed
from library.utils import compute_rmsle


def main():
    print("Starting demonstration of CR-WDS library...")

    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Demonstrate GeometryParser
    print("\n--- Testing GeometryParser ---")
    # We pick the first file from train set based on directory structure
    sample_id = 1
    sample_rel_path = f"train/{sample_id}/geometry.xyz"

    # Verify file exists before parsing (sanity check)
    full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)
    if os.path.exists(full_path):
        atomic_feats, num_atoms, volume = GeometryParser.parse_and_process(
            sample_rel_path
        )

        print(f"Parsed {sample_rel_path}:")
        print(f"  Num atoms: {num_atoms}")
        print(f"  Volume: {volume:.4f}")
        print(f"  Atomic Features Shape: {atomic_feats.shape}")

        # Assertions
        assert atomic_feats.shape == (
            num_atoms,
            Config.ATOMIC_FEATURE_DIM,
        ), f"Expected shape {(num_atoms, Config.ATOMIC_FEATURE_DIM)}, got {atomic_feats.shape}"
        assert volume > 0, "Volume should be positive"
    else:
        print(f"Sample file {sample_rel_path} not found, skipping specific file test.")

    # 3. Demonstrate Model Instantiation and Forward Pass
    print("\n--- Testing Model Forward Pass ---")
    model = CR_WDS().to(device)

    # Create dummy batch
    batch_size = 4
    max_len = 10

    dummy_atomic = torch.randn(batch_size, max_len, Config.ATOMIC_FEATURE_DIM).to(
        device
    )
    dummy_mask = torch.ones(batch_size, max_len, dtype=torch.bool).to(device)
    # Mask out some atoms to test masking mechanism
    dummy_mask[:, -2:] = False
    dummy_global = torch.randn(batch_size, Config.GLOBAL_FEATURE_DIM).to(device)

    batch_dict = {
        "atomic_features": dummy_atomic,
        "mask": dummy_mask,
        "global_features": dummy_global,
    }

    with torch.no_grad():
        outputs = model(batch_dict)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        batch_size,
        Config.NUM_TARGETS,
    ), f"Expected output shape {(batch_size, Config.NUM_TARGETS)}, got {outputs.shape}"

    # 4. Demonstrate Full Training Pipeline (Mini-run)
    print("\n--- Testing Full Training Pipeline (Mini-run) ---")

    # Configuration overrides for speed
    DEBUG_EPOCHS = 2
    DEBUG_BATCH_SIZE = 8
    DEBUG_SAMPLE_SIZE = 50  # Small subset for speed

    # Clean up working directory cache to ensure fresh processing for this demo
    # This ensures we actually test the data processing pipeline
    for f in [
        Config.TRAIN_DATA_CACHE,
        Config.VAL_DATA_CACHE,
        Config.TEST_DATA_CACHE,
        Config.SCALERS_CACHE,
    ]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Generating DataLoaders with debug_size={DEBUG_SAMPLE_SIZE}...")
    # load_cached_data=False forces reprocessing using the debug_size
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=DEBUG_BATCH_SIZE,
        load_cached_data=False,
        debug_size=DEBUG_SAMPLE_SIZE,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize Trainer
    trainer = Trainer(model, device)

    # Train
    print("Fitting model...")
    trainer.fit(train_loader, val_loader, epochs=DEBUG_EPOCHS, patience=1)

    # Predict
    print("Predicting on test set...")
    # Load best model (saved during fit)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            "Warning: No best model saved (likely due to short training), using current weights."
        )

    ids, predictions = trainer.predict(test_loader)

    print(f"Predictions generated for {len(ids)} samples.")
    print(f"Prediction shape: {predictions.shape}")
    if len(predictions) > 0:
        print(f"Sample prediction (ID={ids[0]}): {predictions[0]}")

    # Assertions
    assert len(ids) == len(
        test_loader.dataset
    ), "Number of predictions matches test set size"
    assert predictions.shape == (len(ids), 2), "Prediction shape matches (N, 2)"
    assert (
        predictions >= 0
    ).all(), "Predictions should be non-negative (physical energy)"

    # 5. Generate Submission
    print("\n--- Generating Submission File ---")
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )
    submission_df.sort_values("id", inplace=True)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
