import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import get_data
from library.model import SERVN
from library.training import Trainer, optimize_threshold, predict


def setup_demo_environment():
    """
    Creates a lightweight subset of the data to demonstrate the pipeline quickly.
    It samples a few game_plays from the metadata and filters the raw CSVs accordingly.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_input_dir = "./working/demo_input"
    demo_meta_dir = "./working/demo_metadata"
    demo_working_dir = "./working/demo_working"

    os.makedirs(demo_input_dir, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_working_dir, exist_ok=True)

    # --- Sample Metadata ---
    # Load original metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Select a few game_plays for training/validation to keep it fast.
    # We prioritize plays with contact events to ensure positive samples for training.
    train_gps = train_meta[train_meta["contact"] == 1]["game_play"].unique()[:3]
    if len(train_gps) < 3:
        train_gps = train_meta["game_play"].unique()[:3]

    val_gps = val_meta["game_play"].unique()[:1]
    test_gps = test_meta["game_play"].unique()[:1]

    # Filter metadata
    demo_train_meta = train_meta[train_meta["game_play"].isin(train_gps)].copy()
    demo_val_meta = val_meta[val_meta["game_play"].isin(val_gps)].copy()
    demo_test_meta = test_meta[test_meta["game_play"].isin(test_gps)].copy()

    # Save demo metadata
    demo_train_meta.to_csv(os.path.join(demo_meta_dir, "train.csv"), index=False)
    demo_val_meta.to_csv(os.path.join(demo_meta_dir, "validation.csv"), index=False)
    demo_test_meta.to_csv(os.path.join(demo_meta_dir, "test.csv"), index=False)

    print(f"Demo Train Samples: {len(demo_train_meta)}")
    print(f"Demo Val Samples: {len(demo_val_meta)}")

    # --- Sample Raw Data ---
    # Helper to filter and save raw data files based on selected game_plays
    def filter_and_save(filename, gps, target_dir):
        source_path = os.path.join(Config.INPUT_DIR, filename)
        if not os.path.exists(source_path):
            print(f"Warning: {filename} not found, skipping.")
            return

        # Read full file (fits in memory for this task) and filter
        df = pd.read_csv(source_path)

        if "game_play" in df.columns:
            df_sub = df[df["game_play"].isin(gps)].copy()
            df_sub.to_csv(os.path.join(target_dir, filename), index=False)
        else:
            print(f"Skipping {filename} (no game_play col)")

    print("Filtering raw tracking and helmet data...")

    # Train files (used for both train and val in get_data logic)
    train_val_gps = np.concatenate([train_gps, val_gps])
    filter_and_save("train_player_tracking.csv", train_val_gps, demo_input_dir)
    filter_and_save("train_baseline_helmets.csv", train_val_gps, demo_input_dir)

    # Test files
    filter_and_save("test_player_tracking.csv", test_gps, demo_input_dir)
    filter_and_save("test_baseline_helmets.csv", test_gps, demo_input_dir)

    return demo_input_dir, demo_meta_dir, demo_working_dir


def run_demo():
    # 1. Setup Environment and Patch Config
    # We create a small dataset to ensure the demo runs quickly
    input_dir, meta_dir, working_dir = setup_demo_environment()

    # Monkey-patch Config to use our demo directories and faster training settings
    Config.INPUT_DIR = input_dir
    Config.METADATA_DIR = meta_dir
    Config.WORKING_DIR = working_dir
    Config.EPOCHS = 2  # Reduce epochs for demo
    Config.BATCH_SIZE = 256  # Adjust batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo execution

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # 2. Data Processing
    print("\n=== Data Processing ===")
    # load_cached=False forces the pipeline to process our new demo CSVs
    # instead of loading potentially existing parquet files from a previous run.
    train_dataset, val_dataset, test_dataset, dims, test_ids = get_data(
        load_cached=False
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    print(f"Feature Dimensions: {dims}")

    # Validation checks
    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(val_dataset) > 0, "Validation dataset is empty"
    assert len(test_dataset) > 0, "Test dataset is empty"

    # 3. Model Initialization
    print("\n=== Model Initialization ===")
    model = SERVN(
        kin_input_dim=dims["kin_input_dim"],
        vis_input_dim=dims["vis_input_dim"],
        gate_input_dim=dims["gate_input_dim"],
        num_pos=dims["num_pos"],
        num_team=dims["num_team"],
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass with a single batch
    sample_batch = train_dataset[0]
    # Add batch dimension and move to device
    x_kin = sample_batch["x_kin"].unsqueeze(0).to(Config.DEVICE)
    x_vis = sample_batch["x_vis"].unsqueeze(0).to(Config.DEVICE)
    x_gate = sample_batch["x_gate"].unsqueeze(0).to(Config.DEVICE)
    x_pos = sample_batch["x_pos"].unsqueeze(0).to(Config.DEVICE)
    x_team = sample_batch["x_team"].unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        out = model(x_kin, x_pos, x_team, x_vis, x_gate)

    print(f"Model Output Shape: {out.shape}")
    assert out.shape == (1, 1), "Model output shape mismatch"

    # 4. Training
    print("\n=== Training ===")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    trainer = Trainer(model, train_loader, val_loader, optimizer, Config.DEVICE)
    trainer.fit(epochs=Config.EPOCHS)

    # Verify model saving
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Best model file was not saved"
    print("Training completed and model saved.")

    # 5. Threshold Optimization
    print("\n=== Threshold Optimization ===")
    # Get validation predictions to find optimal threshold
    _, _, y_true, y_pred_probs = trainer.validate()
    best_thresh = optimize_threshold(y_true, y_pred_probs)

    # 6. Inference & Submission
    print("\n=== Inference ===")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Generate probabilities
    test_probs = predict(model, test_loader, Config.DEVICE)

    # Apply threshold
    test_preds = (test_probs > best_thresh).astype(int)

    print(f"Test Predictions Generated: {len(test_preds)}")
    assert len(test_preds) == len(
        test_ids
    ), "Mismatch between predictions and contact IDs"

    # Create submission dataframe
    submission = pd.DataFrame({"contact_id": test_ids, "contact": test_preds})

    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print("Sample Output:")
    print(submission.head())


if __name__ == "__main__":
    run_demo()
