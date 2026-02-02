import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.data import process_data, MaterialDataset, CollateFn, get_datasets
from library.model import SIRDS_SP
from library.train import Trainer
from library.predict import generate_submission
from library.utils import set_seed


def create_dummy_metadata(source_dir, target_dir, num_samples=10):
    """
    Creates dummy metadata CSVs by taking the first few rows from the original metadata.
    This allows us to run the pipeline quickly on real data samples.
    """
    os.makedirs(target_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        source_path = os.path.join(source_dir, f"{split}.csv")
        target_path = os.path.join(target_dir, f"{split}.csv")

        if os.path.exists(source_path):
            df = pd.read_csv(source_path)
            # Take a subset
            dummy_df = df.head(num_samples).copy()
            dummy_df.to_csv(target_path, index=False)
            print(
                f"Created dummy {split} metadata at {target_path} with {len(dummy_df)} samples."
            )
        else:
            print(f"Warning: Source file {source_path} not found.")


def main():
    print("Initializing Demonstration...")

    # 1. Setup & Configuration Monkeypatching
    # We modify Config paths to use a temporary directory for this demo
    # to avoid overwriting real cache files and to speed up processing.
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Create dummy metadata
    create_dummy_metadata(
        Config.METADATA_DIR, os.path.join(DEMO_DIR, "metadata"), num_samples=16
    )

    # Patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    Config.TRAIN_META_PATH = os.path.join(Config.METADATA_DIR, "train.csv")
    Config.VAL_META_PATH = os.path.join(Config.METADATA_DIR, "val.csv")
    Config.TEST_META_PATH = os.path.join(Config.METADATA_DIR, "test.csv")

    Config.TRAIN_DATA_CACHE = os.path.join(DEMO_DIR, "cache", "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(DEMO_DIR, "cache", "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(DEMO_DIR, "cache", "test_data.npz")

    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission", "demo_submission.csv")

    # Reduce training parameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2

    # Ensure cache dir exists
    os.makedirs(os.path.dirname(Config.TRAIN_DATA_CACHE), exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Processing and Dataset Creation
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Data Processing ---")
    # This will process the dummy CSVs we created
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Verify Dataset __getitem__
    sample = train_dataset[0]
    print("Sample keys:", sample.keys())

    # Check shapes
    # Atomic features: (N_atoms, 8) -> 4 one-hot + 3 coords + 1 nn_dist
    assert (
        sample["atom_features"].shape[1] == 8
    ), f"Expected 8 atomic features, got {sample['atom_features'].shape[1]}"
    # Global features: (12,)
    assert (
        sample["global_features"].shape[0] == 12
    ), f"Expected 12 global features, got {sample['global_features'].shape[0]}"
    # Targets: (2,)
    assert (
        sample["target"].shape[0] == 2
    ), f"Expected 2 targets, got {sample['target'].shape[0]}"

    print("Dataset verification successful.")

    # -------------------------------------------------------------------------
    # 3. DataLoader and Collate Function
    # -------------------------------------------------------------------------
    print("\n--- Step 3: DataLoader & Collate ---")
    collate_fn = CollateFn()
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print(f"Batch atom_features shape: {batch['atom_features'].shape}")
    print(f"Batch global_features shape: {batch['global_features'].shape}")
    print(f"Batch targets shape: {batch['targets'].shape}")

    # Verify batch sizes
    assert batch["global_features"].shape[0] == Config.BATCH_SIZE
    assert batch["targets"].shape[0] == Config.BATCH_SIZE
    # Verify packed atoms match batch indices
    assert batch["atom_features"].shape[0] == batch["batch_indices"].shape[0]

    print("DataLoader verification successful.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Model Initialization ---")
    model = SIRDS_SP().to(device)

    # Move batch to device
    atom_features = batch["atom_features"].to(device)
    batch_indices = batch["batch_indices"].to(device)
    global_features = batch["global_features"].to(device)
    spacegroups = batch["spacegroups"].to(device)

    # Forward pass
    outputs = model(atom_features, batch_indices, global_features, spacegroups)
    print(f"Model output shape: {outputs.shape}")

    assert outputs.shape == (Config.BATCH_SIZE, 2), "Model output shape mismatch"
    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Training Loop ---")
    trainer = Trainer(model, device)

    # Run training for limited epochs (Config.EPOCHS = 2)
    trainer.fit(
        train_loader, train_loader
    )  # Using train_loader as val just for demo speed

    # Verify model file was created
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint saved at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Inference ---")
    # Generate submission using the trained model on the dummy test set
    generate_submission(
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        device=device,
    )

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created with {len(df_sub)} rows.")
        assert len(df_sub) == len(
            test_dataset
        ), "Submission row count does not match test dataset size"
        assert "id" in df_sub.columns
        assert "formation_energy_ev_natom" in df_sub.columns
        assert "bandgap_energy_ev" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
