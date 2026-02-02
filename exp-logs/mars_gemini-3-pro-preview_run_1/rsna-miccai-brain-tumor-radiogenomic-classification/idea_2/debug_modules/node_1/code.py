import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
)
from library.utils import set_seed
from library.dataset import prepare_data_cache, RSNADataset, get_transforms
from library.model import MILNet
from library.engine import run_training, generate_submission

if __name__ == "__main__":
    # 1. Setup and Reproducibility
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Load Metadata
    # We load the metadata CSVs generated previously.
    print("Loading metadata...")
    df_train_full = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(VAL_METADATA_PATH)
    df_test_full = pd.read_csv(TEST_METADATA_PATH)

    # 3. Optimize for Speed: Create small subsets
    # Processing thousands of DICOM files takes time. For this demo, we subset the data.
    SUBSET_SIZE_TRAIN = 16
    SUBSET_SIZE_VAL = 8
    SUBSET_SIZE_TEST = 8

    df_train_subset = df_train_full.head(SUBSET_SIZE_TRAIN).copy()
    df_val_subset = df_val_full.head(SUBSET_SIZE_VAL).copy()
    df_test_subset = df_test_full.head(SUBSET_SIZE_TEST).copy()

    print(
        f"Subset sizes -> Train: {len(df_train_subset)}, Val: {len(df_val_subset)}, Test: {len(df_test_subset)}"
    )

    # 4. Prepare Data Caches
    # We generate temporary cache files for these subsets in the working directory.
    # This scans the input directory to find valid DICOM slices.
    train_cache_file = os.path.join(WORKING_DIR, "demo_train.parquet")
    val_cache_file = os.path.join(WORKING_DIR, "demo_val.parquet")
    test_cache_file = os.path.join(WORKING_DIR, "demo_test.parquet")

    print("Preprocessing and caching file paths...")
    df_train_processed = prepare_data_cache(
        df_train_subset, train_cache_file, load_cached_data=False
    )
    df_val_processed = prepare_data_cache(
        df_val_subset, val_cache_file, load_cached_data=False
    )
    df_test_processed = prepare_data_cache(
        df_test_subset, test_cache_file, load_cached_data=False
    )

    # 5. Instantiate Datasets and Loaders
    # We use the transforms defined in the library.
    train_dataset = RSNADataset(df_train_processed, transform=get_transforms("train"))
    val_dataset = RSNADataset(df_val_processed, transform=get_transforms("valid"))
    test_dataset = RSNADataset(df_test_processed, transform=get_transforms("test"))

    # Verify dataset output shape
    # Expected shape: (NUM_SLICES, CHANNELS, H, W) -> (32, 3, 256, 256) based on config
    sample_data, sample_target = train_dataset[0]
    assert (
        sample_data.dim() == 4
    ), f"Expected 4D tensor (N, C, H, W), got {sample_data.shape}"
    assert sample_data.shape[1] == 3, f"Expected 3 channels, got {sample_data.shape[1]}"
    print("Dataset verification passed.")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 6. Initialize Model
    print("Initializing MILNet...")
    model = MILNet().to(DEVICE)

    # 7. Run Training
    # We override NUM_EPOCHS to 2 for a quick demonstration.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    print("Starting training loop...")
    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=2,  # Reduced for demo speed
        patience=2,
        device=DEVICE,
    )

    # 8. Generate Submission
    submission_file = os.path.join(WORKING_DIR, "demo_submission.csv")
    generate_submission(trained_model, test_loader, submission_file, DEVICE)

    # 9. Verify Submission
    print("Verifying submission file...")
    assert os.path.exists(submission_file), "Submission file not found."

    df_sub = pd.read_csv(submission_file)

    # Check dimensions
    assert len(df_sub) == len(
        df_test_subset
    ), f"Submission length {len(df_sub)} mismatch with test set {len(df_test_subset)}"

    # Check columns
    assert "BraTS21ID" in df_sub.columns, "BraTS21ID column missing"
    assert "MGMT_value" in df_sub.columns, "MGMT_value column missing"

    # Check ID matching
    expected_ids = df_test_subset["BraTS21ID"].values
    actual_ids = df_sub["BraTS21ID"].values
    # Sorting might differ depending on loader, but set content should be same
    assert set(expected_ids) == set(
        actual_ids
    ), "Submission IDs do not match test set IDs"

    # Check probability range
    probs = df_sub["MGMT_value"].values
    assert (probs >= 0.0).all() and (
        probs <= 1.0
    ).all(), "Predictions contain values outside [0, 1]"

    print("Demo completed successfully. All checks passed.")
