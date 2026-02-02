import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
import library.config as config
import library.data_utils as data_utils
import library.dataset as dataset
import library.model as model_lib
import library.train as train_lib
import library.predict as predict_lib


def setup_demo_environment():
    """
    Sets up a temporary directory for the demo and creates subset CSVs
    to ensure the demonstration runs quickly.
    """
    base_dir = "./working/demo_execution"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    print(f"[Demo] Created working directory: {base_dir}")

    # Load original metadata
    train_full = pd.read_csv(config.TRAIN_CSV)
    val_full = pd.read_csv(config.VAL_CSV)
    test_full = pd.read_csv(config.TEST_CSV)

    # Create subsets (e.g., 10 samples each)
    n_samples = 10
    train_subset = train_full.head(n_samples).copy()
    val_subset = val_full.head(n_samples).copy()
    test_subset = test_full.head(n_samples).copy()

    # Save subsets
    train_subset_path = os.path.join(base_dir, "train.csv")
    val_subset_path = os.path.join(base_dir, "val.csv")
    test_subset_path = os.path.join(base_dir, "test.csv")
    submission_path = os.path.join(base_dir, "demo_submission.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    print(f"[Demo] Created subset metadata files with {n_samples} samples each.")

    # Monkey-patch the library modules to use our demo paths and directory
    # This redirects the library to use our small datasets and temporary output dir

    # Patch config (though modules might have already imported values)
    config.WORKING_DIR = base_dir

    # Patch data_utils
    data_utils.WORKING_DIR = base_dir

    # Patch train module variables
    train_lib.WORKING_DIR = base_dir
    train_lib.TRAIN_CSV = train_subset_path
    train_lib.VAL_CSV = val_subset_path

    # Patch predict module variables
    predict_lib.WORKING_DIR = base_dir
    predict_lib.TEST_CSV = test_subset_path
    predict_lib.SUBMISSION_FILE = submission_path

    return train_subset_path, val_subset_path, test_subset_path


def test_data_processing(train_csv_path):
    print("\n" + "=" * 40)
    print("Testing Data Processing (data_utils.py)")
    print("=" * 40)

    # Test process_dataset
    # load_cached_data=False ensures we actually run the processing logic
    data_dict = data_utils.process_dataset(train_csv_path, load_cached_data=False)

    # Verification
    assert "atomic_features" in data_dict
    assert "global_features" in data_dict
    assert "targets" in data_dict
    assert "ids" in data_dict

    n_samples = len(data_dict["ids"])
    print(f"[Check] Processed {n_samples} samples.")
    assert n_samples == 10, f"Expected 10 samples, got {n_samples}"

    # Check shapes
    # Global features dim should be 15 as per data_utils code
    assert (
        data_dict["global_features"].shape[1] == 15
    ), "Global features dimension mismatch"
    # Targets dim should be 2
    assert data_dict["targets"].shape[1] == 2, "Targets dimension mismatch"

    # Test get_scalers
    scalers = data_utils.get_scalers(data_dict)
    print(f"[Check] Scalers computed: {list(scalers.keys())}")
    assert "global_mean" in scalers
    assert "global_std" in scalers

    return scalers


def test_dataset_and_collate(train_csv_path, scalers):
    print("\n" + "=" * 40)
    print("Testing Dataset and Collate (dataset.py)")
    print("=" * 40)

    # Initialize Dataset
    ds = dataset.MaterialDataset(train_csv_path, scalers=scalers, load_cached_data=True)
    print(f"[Check] Dataset length: {len(ds)}")

    # Get one item
    item = ds[0]
    print(f"[Check] Item keys: {list(item.keys())}")
    assert isinstance(item["atomic_features"], torch.Tensor)
    assert isinstance(item["global_features"], torch.Tensor)

    # Test DataLoader with Collate
    dl = DataLoader(ds, batch_size=4, collate_fn=dataset.collate_materials)
    batch = next(iter(dl))

    print(f"[Check] Batch keys: {list(batch.keys())}")
    # Check padding mask
    assert "mask" in batch
    # Check atomic features shape: (Batch, Max_Atoms, Feat_Dim)
    # Feat dim is 8 (4 onehot + 3 coords + 1 dist)
    assert batch["atomic_features"].dim() == 3
    assert batch["atomic_features"].shape[2] == 8
    assert batch["atomic_features"].shape[0] == 4

    print("[Check] Batch collation successful.")
    return batch


def test_model_architecture(batch):
    print("\n" + "=" * 40)
    print("Testing Model Architecture (model.py)")
    print("=" * 40)

    atomic_dim = batch["atomic_features"].shape[2]
    global_dim = batch["global_features"].shape[1]

    # Instantiate model
    model = model_lib.PIGWDS(
        atomic_input_dim=atomic_dim,
        global_input_dim=global_dim,
        atomic_hidden_dim=32,  # Reduced for demo
        global_hidden_dim=16,
        fusion_hidden_dim=16,
        output_dim=2,
        dropout=0.0,
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(
            batch["atomic_features"], batch["global_features"], batch["mask"]
        )

    print(f"[Check] Output shape: {output.shape}")
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"
    print("[Check] Forward pass successful.")


def test_training_loop():
    print("\n" + "=" * 40)
    print("Testing Training Loop (train.py)")
    print("=" * 40)

    # Run training for 1 epoch
    # Note: We patched the CSV paths in setup_demo_environment
    train_lib.train_model(
        epochs=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        patience=1,
        load_cached_data=True,  # Should use the cache generated in test_data_processing
    )

    # Check if artifacts were created
    model_path = os.path.join(train_lib.WORKING_DIR, "best_model.pt")
    scalers_path = os.path.join(train_lib.WORKING_DIR, "scalers.npz")

    if os.path.exists(model_path):
        print(f"[Check] Model saved at {model_path}")
    else:
        raise AssertionError("Model file was not saved.")

    if os.path.exists(scalers_path):
        print(f"[Check] Scalers saved at {scalers_path}")
    else:
        raise AssertionError("Scalers file was not saved.")


def test_inference():
    print("\n" + "=" * 40)
    print("Testing Inference (predict.py)")
    print("=" * 40)

    # Run prediction
    # Note: We patched TEST_CSV and SUBMISSION_FILE in setup_demo_environment
    predict_lib.generate_predictions(
        batch_size=4, load_cached_data=False  # Force processing of test subset
    )

    submission_file = predict_lib.SUBMISSION_FILE
    if os.path.exists(submission_file):
        df = pd.read_csv(submission_file)
        print(f"[Check] Submission generated with {len(df)} rows.")
        assert len(df) == 10, "Submission should have 10 rows matching test subset."
        assert "formation_energy_ev_natom" in df.columns
        assert "bandgap_energy_ev" in df.columns
    else:
        raise AssertionError("Submission file was not generated.")


def main():
    # Set seed for reproducibility
    train_lib.set_seed(42)

    # 1. Setup
    train_csv, val_csv, test_csv = setup_demo_environment()

    # 2. Data Processing
    scalers = test_data_processing(train_csv)

    # 3. Dataset & Collate
    batch = test_dataset_and_collate(train_csv, scalers)

    # 4. Model
    test_model_architecture(batch)

    # 5. Training
    test_training_loop()

    # 6. Inference
    test_inference()

    print("\n" + "=" * 40)
    print("All demonstrations completed successfully.")
    print("=" * 40)


if __name__ == "__main__":
    main()
