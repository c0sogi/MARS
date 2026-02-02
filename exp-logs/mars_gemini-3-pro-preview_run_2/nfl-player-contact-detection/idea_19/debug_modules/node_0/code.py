import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, FocalLoss, optimize_mcc_threshold
from library.data_processing import process_data
from library.dataset import get_dataloader
from library.models import SRVNet
from library.train import train_one_epoch, evaluate
from library.inference import run_inference


def main():
    print("=== Starting SRV-Net Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Set deterministic seed
    set_seed(42)

    # Define temporary directories for the demo
    DEMO_WORKING_DIR = "./working/demo_run/working"
    DEMO_META_DIR = "./working/demo_run/metadata"

    # Clean up previous runs if they exist
    if os.path.exists("./working/demo_run"):
        shutil.rmtree("./working/demo_run")

    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Override Config class attributes for the demo
    # We point metadata to our subset folder and working to our demo folder
    Config.METADATA_DIR = DEMO_META_DIR
    Config.WORKING_DIR = DEMO_WORKING_DIR

    # Reduce compute requirements for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # -------------------------------------------------------------------------
    # 2. Create Data Subset (Mocking Metadata)
    # -------------------------------------------------------------------------
    print("\n[2] Creating data subset for rapid execution...")

    # We read the original metadata and sample 1 game_play for train/val/test
    # This forces process_data to only load tracking data for these specific plays.

    orig_meta_dir = "./metadata"

    for split in ["train", "validation", "test"]:
        orig_path = os.path.join(orig_meta_dir, f"{split}.csv")
        df = pd.read_csv(orig_path)

        # Pick the first unique game_play
        target_gp = df["game_play"].unique()[0]
        subset_df = df[df["game_play"] == target_gp].copy()

        # Save to demo metadata directory
        save_path = os.path.join(DEMO_META_DIR, f"{split}.csv")
        subset_df.to_csv(save_path, index=False)
        print(
            f"    Created {split} subset with {len(subset_df)} rows (GamePlay: {target_gp})"
        )

    # -------------------------------------------------------------------------
    # 3. Data Processing Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Running Data Processing Pipeline...")

    # Process Train
    # load_cached_data=False forces re-processing from our new subset metadata
    X_kin_train, X_vis_train, y_train, ids_train = process_data(
        "train", load_cached_data=False
    )

    # Process Validation
    X_kin_val, X_vis_val, y_val, ids_val = process_data(
        "validation", load_cached_data=False
    )

    # Verify Shapes
    # Kinematic Dim: 19 features * 11 steps = 209
    # Visual Dim: 14 features * 11 steps = 154
    print(f"    Train Kinematic Shape: {X_kin_train.shape}")
    print(f"    Train Visual Shape:    {X_vis_train.shape}")
    print(f"    Train Targets Shape:   {y_train.shape}")

    assert (
        X_kin_train.shape[1] == Config.INPUT_DIM_KINEMATIC
    ), f"Kinematic feature dim mismatch. Expected {Config.INPUT_DIM_KINEMATIC}, got {X_kin_train.shape[1]}"
    assert (
        X_vis_train.shape[1] == Config.INPUT_DIM_VISUAL
    ), f"Visual feature dim mismatch. Expected {Config.INPUT_DIM_VISUAL}, got {X_vis_train.shape[1]}"
    assert len(X_kin_train) == len(y_train), "Feature/Target length mismatch"

    # -------------------------------------------------------------------------
    # 4. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[4] Initializing DataLoaders...")

    train_loader = get_dataloader(
        X_kin_train,
        X_vis_train,
        y_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = get_dataloader(
        X_kin_val,
        X_vis_val,
        y_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify batch structure
    sample_kin, sample_vis, sample_y = next(iter(train_loader))
    print(f"    Batch Kinematic Shape: {sample_kin.shape}")
    print(f"    Batch Visual Shape:    {sample_vis.shape}")
    print(f"    Batch Label Shape:     {sample_y.shape}")

    assert sample_kin.shape == (Config.BATCH_SIZE, Config.INPUT_DIM_KINEMATIC)
    assert sample_y.shape == (Config.BATCH_SIZE, 1)

    # -------------------------------------------------------------------------
    # 5. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[5] Initializing SRVNet Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    model = SRVNet(
        input_dim_kin=Config.INPUT_DIM_KINEMATIC,
        input_dim_vis=Config.INPUT_DIM_VISUAL,
        kinematic_hidden_dims=Config.KINEMATIC_HIDDEN_DIMS,
        visual_hidden_dims=Config.VISUAL_HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT_RATE,
        lambda_visual=Config.LAMBDA_VISUAL,
    ).to(device)

    # Verify Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(sample_kin.to(device), sample_vis.to(device))

    print(f"    Output Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape incorrect"

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA)

    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")

    # Evaluate
    val_loss, val_mcc, val_thresh = evaluate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val MCC:  {val_mcc:.4f} (at threshold {val_thresh:.2f})")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # -------------------------------------------------------------------------
    # 7. Inference & Threshold Optimization
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference & Optimization...")

    # Run inference on validation set again to demonstrate run_inference utility
    val_probs = run_inference(model, val_loader, device)

    # Verify probabilities range
    assert (
        val_probs.min() >= 0.0 and val_probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    # Optimize threshold
    best_thresh, best_score = optimize_mcc_threshold(y_val, val_probs)
    print(f"    Optimized Threshold: {best_thresh:.4f}")
    print(f"    Optimized MCC:       {best_score:.4f}")

    # Save dummy model for completeness
    torch.save(model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth"))
    print("    Model saved.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
