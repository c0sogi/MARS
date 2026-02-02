import os
import torch
import pandas as pd
import numpy as np
import warnings
from library.config import Config
from library.utils import seed_everything, compute_mcrmse
from library.data import get_dataloaders
from library.model import SDBR_BiGRU
from library.engine import Engine

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("==== RNA Degradation Prediction Demo ====")

    # 1. Setup and Reproducibility
    seed_everything(42)

    # Define working directory for this demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")

    # 2. Create Mini Datasets for Speed
    # We load the original metadata and slice it to create a tiny dataset
    print("\n[1/6] Creating mini datasets for rapid demonstration...")

    try:
        # Load original metadata
        df_train_full = pd.read_parquet("./metadata/train.parquet")
        df_val_full = pd.read_parquet("./metadata/val.parquet")
        df_test_full = pd.read_parquet("./metadata/test.parquet")

        # Slice (20 train, 10 val, 10 test)
        df_train_mini = df_train_full.head(20)
        df_val_mini = df_val_full.head(10)
        df_test_mini = df_test_full.head(10)

        # Save mini datasets
        train_mini_path = os.path.join(demo_dir, "train_subset.parquet")
        val_mini_path = os.path.join(demo_dir, "val_subset.parquet")
        test_mini_path = os.path.join(demo_dir, "test_subset.parquet")

        df_train_mini.to_parquet(train_mini_path)
        df_val_mini.to_parquet(val_mini_path)
        df_test_mini.to_parquet(test_mini_path)

        print(f"  Train subset: {len(df_train_mini)} samples")
        print(f"  Val subset:   {len(df_val_mini)} samples")
        print(f"  Test subset:  {len(df_test_mini)} samples")

    except Exception as e:
        print(f"Error creating subsets: {e}")
        return

    # 3. Override Configuration
    print("\n[2/6] Configuring environment...")

    # Point Config to the new mini files
    Config.TRAIN_PATH = train_mini_path
    Config.VAL_PATH = val_mini_path
    Config.TEST_PATH = test_mini_path

    # Point Cache to the demo directory to avoid conflicts
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_cache.npz")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_cache.npz")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_cache.npz")

    # Set output paths
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")
    Config.WORKING_DIR = demo_dir

    # Optimize hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # 4. Data Loading and Verification
    print("\n[3/6] Initializing DataLoaders...")

    # Force reprocessing to ensure we use the mini datasets (load_cached_data=False)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    features = sample_batch["features"]
    pair_indices = sample_batch["pair_indices"]
    targets = sample_batch["targets"]

    print(f"  Batch Features Shape: {features.shape} (Expected: [4, 107, 14])")
    print(f"  Batch Targets Shape:  {targets.shape} (Expected: [4, 107, 5])")

    assert features.shape == (4, 107, 14), "Feature shape mismatch"
    assert targets.shape == (4, 107, 5), "Target shape mismatch"
    assert pair_indices.shape == (4, 107), "Pair indices shape mismatch"
    print("  Data integrity verified.")

    # 5. Model Initialization and Forward Pass Verification
    print("\n[4/6] Initializing Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = SDBR_BiGRU()
    model.to(device)

    # Run a dummy forward pass
    with torch.no_grad():
        dummy_feats = features.to(device)
        dummy_idx = pair_indices.to(device)
        dummy_mask = sample_batch["pair_masks"].to(device)

        output = model(dummy_feats, dummy_idx, dummy_mask)

    print(f"  Model Output Shape: {output.shape} (Expected: [4, 107, 5])")
    assert output.shape == (4, 107, 5), "Model output shape mismatch"
    print("  Model forward pass verified.")

    # 6. Metric Logic Verification
    print("\n[5/6] Verifying Metric Calculation Logic...")

    # Create controlled dummy data
    # Shape: (1 sample, 107 length, 5 targets)
    # Scored targets are indices [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Scored length is 68
    t_dummy = np.zeros((1, 107, 5))
    p_dummy = np.zeros((1, 107, 5))

    # Introduce a known error:
    # Target 0 (reactivity) at position 0: True=1.0, Pred=0.0 -> Error=1.0
    # RMSE for Target 0 = sqrt(1/68 * 1^2) = sqrt(1/68) ≈ 0.1212678
    # RMSE for Target 1 = 0.0
    # RMSE for Target 3 = 0.0
    # MCRMSE = (0.1212678 + 0 + 0) / 3 ≈ 0.0404226
    t_dummy[0, 0, 0] = 1.0
    p_dummy[0, 0, 0] = 0.0

    calculated_score = compute_mcrmse(p_dummy, t_dummy)
    expected_score = (np.sqrt(1 / 68)) / 3

    print(f"  Calculated MCRMSE: {calculated_score:.6f}")
    print(f"  Expected MCRMSE:   {expected_score:.6f}")

    # Allow small floating point tolerance
    assert (
        abs(calculated_score - expected_score) < 1e-5
    ), "Metric calculation logic failed"
    print("  Metric logic verified.")

    # 7. Training and Inference Loop
    print("\n[6/6] Running Training and Inference...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    engine = Engine(model, optimizer, device=device)

    # Train
    engine.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Predict
    print("  Generating predictions on test set...")
    preds, ids = engine.predict(test_loader)

    assert len(preds) == len(df_test_mini), "Prediction count mismatch"
    assert preds.shape == (
        len(df_test_mini),
        107,
        5,
    ), "Prediction tensor shape mismatch"

    # Format Submission
    print("  Formatting submission...")
    preds_flat = preds.reshape(-1, 5)

    # Generate ID_seqpos keys
    id_seqpos = []
    for sample_id in ids:
        for i in range(107):
            id_seqpos.append(f"{sample_id}_{i}")

    submission_df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", id_seqpos)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"  Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"  Submission shape: {submission_df.shape}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
