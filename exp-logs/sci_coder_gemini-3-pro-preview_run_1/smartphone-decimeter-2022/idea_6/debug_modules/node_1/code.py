import os
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

# Import library modules
import library.config as C
import library.utils as U
import library.data_processing as DP
import library.dataset as DS
import library.model as M
import library.engine as E


def create_subset_metadata(src_filename, dst_filename, n_rows_limit=5000):
    """
    Creates a smaller metadata CSV containing only the first available drive.
    This ensures we process a coherent sequence quickly.
    """
    src_path = os.path.join(C.METADATA_DIR, src_filename)
    dst_path = os.path.join(C.WORKING_DIR, dst_filename)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source metadata file not found: {src_path}")

    df = pd.read_csv(src_path)

    if df.empty:
        print(f"Warning: {src_filename} is empty.")
        df.to_csv(dst_path, index=False)
        return dst_path, 0

    # Filter for a single drive to maintain sequence integrity
    if "drive_id" in df.columns:
        unique_drives = df["drive_id"].unique()
        if len(unique_drives) > 0:
            target_drive = unique_drives[0]
            df = df[df["drive_id"] == target_drive]
            print(f"Selected drive '{target_drive}' for {dst_filename}")

    # Further limit rows if the drive is huge
    if len(df) > n_rows_limit:
        df = df.iloc[:n_rows_limit]

    df.to_csv(dst_path, index=False)
    return dst_path, len(df)


def main():
    print("=== Starting GNSS Location Prediction Demo ===")

    # 1. Configure for Speed
    # Override config settings to run a minimal training loop
    C.NUM_EPOCHS = 1
    C.BATCH_SIZE = 2
    C.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure working directory exists
    os.makedirs(C.WORKING_DIR, exist_ok=True)

    # 2. Prepare Data Subsets
    print("\n[Step 1] Preparing Metadata Subsets...")
    train_meta_path, n_train = create_subset_metadata(
        "train_metadata.csv", "train_meta_sub.csv"
    )
    val_meta_path, n_val = create_subset_metadata(
        "val_metadata.csv", "val_meta_sub.csv"
    )
    test_meta_path, n_test = create_subset_metadata(
        "test_metadata.csv", "test_meta_sub.csv"
    )

    print(f"Subset counts -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # 3. Process Sequences
    print("\n[Step 2] Processing GNSS Sequences...")
    # We use load_cached_data=False to force processing logic verification
    # split_name is used to suffix cache files to avoid collisions
    train_seqs = DP.prepare_sequences(
        train_meta_path, load_cached_data=False, split_name="train_sub"
    )
    val_seqs = DP.prepare_sequences(
        val_meta_path, load_cached_data=False, split_name="val_sub"
    )
    test_seqs = DP.prepare_sequences(
        test_meta_path, load_cached_data=False, split_name="test_sub"
    )

    if not train_seqs:
        print("No training data found/processed. Exiting.")
        return

    # 4. Create Datasets
    print("\n[Step 3] Creating PyTorch Datasets...")
    # Train dataset computes normalization stats
    train_dataset = DS.GNSSSequenceDataset(train_seqs, split="train")

    # Val and Test datasets use training stats
    val_dataset = DS.GNSSSequenceDataset(
        val_seqs, split="val", mean=train_dataset.mean, std=train_dataset.std
    )
    test_dataset = DS.GNSSSequenceDataset(
        test_seqs, split="test", mean=train_dataset.mean, std=train_dataset.std
    )

    print(f"Train sequences: {len(train_dataset)}")

    # Verify a sample
    sample = train_dataset[0]
    print(f"Sample feature shape: {sample['features'].shape}")
    print(f"Sample target shape: {sample['targets'].shape}")

    # 5. Create DataLoaders
    print("\n[Step 4] Initializing DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        collate_fn=DS.gnss_collate_fn,
        num_workers=C.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        collate_fn=DS.gnss_collate_fn,
        num_workers=C.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        collate_fn=DS.gnss_collate_fn,
        num_workers=C.NUM_WORKERS,
    )

    # 6. Initialize Model
    print("\n[Step 5] Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = M.TransResUNet().to(device)

    # Verify Forward Pass
    batch = next(iter(train_loader))
    feats = batch["features"].to(device).permute(0, 2, 1)  # (B, C, L)
    phone_idx = batch["phone_idx"].to(device)

    with torch.no_grad():
        out = model(feats, phone_idx)

    print(f"Model input shape: {feats.shape}")
    print(f"Model output shape: {out.shape}")

    # Assert output dimensions match config
    assert (
        out.shape[1] == C.OUTPUT_DIM
    ), f"Expected output dim {C.OUTPUT_DIM}, got {out.shape[1]}"
    assert (
        out.shape[2] == feats.shape[2]
    ), "Temporal dimension mismatch between input and output"

    # 7. Training
    print("\n[Step 6] Running Training Loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY
    )
    trainer = E.Trainer(model, train_loader, val_loader, optimizer, device)

    # Run fit (configured to 1 epoch)
    trainer.fit(epochs=C.NUM_EPOCHS)

    # 8. Inference
    print("\n[Step 7] Generating Submission...")
    E.generate_submission(model, test_loader, device)

    # Verify Submission File
    sub_path = os.path.join(C.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(sub_path):
        df_sub = pd.read_csv(sub_path)
        print(f"Submission file created successfully with {len(df_sub)} rows.")
        print(df_sub.head())

        # Basic validation
        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        for col in required_cols:
            assert col in df_sub.columns, f"Missing column {col} in submission"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
