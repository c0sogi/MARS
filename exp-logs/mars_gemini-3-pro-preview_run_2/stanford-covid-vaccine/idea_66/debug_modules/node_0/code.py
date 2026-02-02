import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.data import process_data, RNADataset
from library.layers import SpatialStem, DenseDilatedBlock, DenseTCN
from library.model import GCSDNModel
from library.train import train_one_epoch, validate


def run_demo():
    print("==== STARTING DEMO SCRIPT ====")

    # 1. SETUP & CONFIGURATION OVERRIDE
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for speed...")

    # Define a temporary directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters to ensure speed
    Config.WORKING_DIR = DEMO_DIR
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.CACHE_TRAIN = "demo_train.npz"
    Config.CACHE_VAL = "demo_val.npz"
    Config.CACHE_TEST = "demo_test.npz"
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. DATA PREPARATION (MINI DATASETS)
    # ---------------------------------------------------------
    print("\n[2] Creating mini-datasets...")

    # Load original metadata
    orig_train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    orig_val_path = os.path.join(Config.METADATA_DIR, "val.csv")
    orig_test_path = os.path.join(Config.METADATA_DIR, "test.csv")

    # Create mini versions (16 samples for train, 8 for val, 8 for test)
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    pd.read_csv(orig_train_path).head(16).to_csv(mini_train_path, index=False)
    pd.read_csv(orig_val_path).head(8).to_csv(mini_val_path, index=False)
    pd.read_csv(orig_test_path).head(8).to_csv(mini_test_path, index=False)

    print("Mini-datasets created.")

    # 3. DATA PROCESSING
    # ---------------------------------------------------------
    print("\n[3] Processing data using library.data.process_data...")

    train_data = process_data(
        mini_train_path, is_test=False, cache_name=Config.CACHE_TRAIN
    )
    val_data = process_data(mini_val_path, is_test=False, cache_name=Config.CACHE_VAL)
    test_data = process_data(mini_test_path, is_test=True, cache_name=Config.CACHE_TEST)

    # Verify data shapes
    # Features should be (N, L, 18) based on library.data logic before dataset conversion
    # Note: library.data returns features as (N, L, 18) numpy array
    assert train_data["features"].shape[1] == Config.SEQ_LENGTH
    assert (
        train_data["features"].shape[2] == 18
    )  # 4 seq + 3 struct + 7 loop + 4 partner
    assert "targets" in train_data
    assert train_data["targets"].shape[1] == 5  # 5 target columns

    print("Data processing verified.")

    # 4. DATASET & DATALOADER
    # ---------------------------------------------------------
    print("\n[4] Initializing Datasets and DataLoaders...")

    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Verify a batch
    feat, p_idx, target = next(iter(train_loader))
    # Dataset permutes features to (C, L) for Conv1d
    assert feat.shape == (Config.BATCH_SIZE, 18, Config.SEQ_LENGTH)
    assert p_idx.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert target.shape == (Config.BATCH_SIZE, 5, Config.SEQ_LENGTH)

    print("DataLoader batch shapes verified.")

    # 5. LAYER VERIFICATION
    # ---------------------------------------------------------
    print("\n[5] Verifying library.layers components...")

    # Test SpatialStem
    stem = SpatialStem(in_channels=18, out_channels=32).to(device)
    dummy_input = torch.randn(2, 18, Config.SEQ_LENGTH).to(device)
    stem_out = stem(dummy_input)
    assert stem_out.shape == (
        2,
        32,
        Config.SEQ_LENGTH,
    ), f"SpatialStem output shape mismatch: {stem_out.shape}"

    # Test DenseDilatedBlock
    block = DenseDilatedBlock(in_channels=32, growth_rate=12, dilation=1).to(device)
    block_out = block(stem_out)
    assert block_out.shape == (
        2,
        12,
        Config.SEQ_LENGTH,
    ), f"DenseDilatedBlock output shape mismatch: {block_out.shape}"

    # Test DenseTCN
    tcn = DenseTCN(in_channels=32, growth_rate=12, dilations=[1, 2]).to(device)
    tcn_out = tcn(stem_out)
    # DenseTCN concatenates input + block outputs.
    # Input(32) + Block1(12) + Block2(12) = 56 channels expected if DenseNet style concatenation happens
    # Checking implementation: DenseTCN returns torch.cat(features, dim=1).
    # Initial features=[x] (32). Block1 adds 12. Block2 adds 12. Total 56.
    expected_channels = 32 + 12 + 12
    assert tcn_out.shape == (
        2,
        expected_channels,
        Config.SEQ_LENGTH,
    ), f"DenseTCN output shape mismatch: {tcn_out.shape}"

    print("Layer components verified.")

    # 6. MODEL VERIFICATION
    # ---------------------------------------------------------
    print("\n[6] Verifying GCSDNModel...")

    model = GCSDNModel().to(device)

    # Run forward pass
    # Model expects (features, partner_indices)
    dummy_pidx = torch.zeros((2, Config.SEQ_LENGTH), dtype=torch.long).to(device)
    y1, y2 = model(dummy_input, dummy_pidx)

    assert y1.shape == (2, 5, Config.SEQ_LENGTH), "Pass 1 output shape mismatch"
    assert y2.shape == (2, 5, Config.SEQ_LENGTH), "Pass 2 output shape mismatch"

    print("Model forward pass verified.")

    # 7. TRAINING LOOP DEMO
    # ---------------------------------------------------------
    print("\n[7] Demonstrating Training Loop (train_one_epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for 1 epoch
    initial_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Training Epoch 1 Loss: {initial_loss:.4f}")

    assert isinstance(initial_loss, float)
    assert not np.isnan(initial_loss)

    print("Training loop execution successful.")

    # 8. VALIDATION DEMO
    # ---------------------------------------------------------
    print("\n[8] Demonstrating Validation (validate)...")

    val_score = validate(model, val_loader, device)
    print(f"Validation MCRMSE: {val_score:.4f}")

    assert isinstance(val_score, float)
    assert val_score >= 0.0

    print("Validation execution successful.")

    # 9. INFERENCE & SUBMISSION DEMO
    # ---------------------------------------------------------
    print("\n[9] Demonstrating Inference and Submission Generation...")

    model.eval()
    preds_map = {}

    with torch.no_grad():
        for features, p_idx, ids in test_loader:
            features = features.to(device)
            p_idx = p_idx.to(device)

            # Use Pass 2 output
            _, y2 = model(features, p_idx)
            y_np = y2.cpu().numpy()  # (B, 5, L)

            for i, sample_id in enumerate(ids):
                for pos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{pos}"
                    preds_map[row_id] = y_np[i, :, pos]

    # Load sample submission to verify mapping
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # We only have predictions for the IDs in our mini_test.csv
    # We will create a submission dataframe just for those to prove it works
    # Or fill the rest with zeros as per the library.train.generate_submission logic

    submission_data = []
    # Just check the first few rows of sample_sub that match our mini test set
    # The mini test set IDs are:
    mini_test_ids = pd.read_csv(mini_test_path)["id"].values

    matched_count = 0
    for _, row in sample_sub.iterrows():
        row_id = row["id_seqpos"]
        # Extract ID from row_id (format: id_xxxxx_seqpos)
        # Simple check: if row_id is in preds_map
        if row_id in preds_map:
            submission_data.append(preds_map[row_id])
            matched_count += 1
        else:
            # Just break early for speed, we proved the point if we matched some
            if matched_count > 0:
                break
            submission_data.append(np.zeros(5))

    print(
        f"Generated predictions for {matched_count} sequence positions (partial check)."
    )

    # Save a dummy submission file to prove file IO works
    # We'll just save the predictions we made
    demo_sub_df = pd.DataFrame(
        list(preds_map.values()),
        columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
    )
    demo_sub_df["id_seqpos"] = list(preds_map.keys())
    # Reorder columns
    demo_sub_df = demo_sub_df[
        ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    ]

    demo_sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Demo submission saved to {Config.SUBMISSION_PATH}")

    print("\n==== DEMO COMPLETED SUCCESSFULLY ====")


if __name__ == "__main__":
    run_demo()
