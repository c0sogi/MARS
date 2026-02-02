import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
import library.feature_engineering as fe
import library.dataset as ds
import library.model as model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting NFL Contact Detection Demo ===\n")

    # 1. Setup and Reproducibility
    print("Step 1: Setting random seeds...")
    utils.seed_everything(config.SEED)

    # 2. Data Loading (Subset for Speed)
    print("Step 2: Loading data subset...")

    # Load metadata and pick a single game_play to keep runtime minimal
    df_meta = pd.read_csv(config.TRAIN_META_PATH)
    selected_game_play = df_meta["game_play"].unique()[0]
    print(f"  Selected GamePlay for demo: {selected_game_play}")

    df_meta_subset = df_meta[df_meta["game_play"] == selected_game_play].copy()

    # Load corresponding tracking data
    # In a real run, we might iterate through chunks, but here we load and filter
    print("  Loading tracking data...")
    df_tracking = pd.read_csv(config.TRAIN_TRACKING_PATH)
    df_tracking_subset = df_tracking[
        df_tracking["game_play"] == selected_game_play
    ].copy()

    # Load corresponding helmet data
    print("  Loading helmet data...")
    df_helmets = pd.read_csv(config.TRAIN_HELMETS_PATH)
    df_helmets_subset = df_helmets[df_helmets["game_play"] == selected_game_play].copy()

    print(
        f"  Subset Sizes -> Labels: {len(df_meta_subset)}, Tracking: {len(df_tracking_subset)}, Helmets: {len(df_helmets_subset)}"
    )

    # 3. Feature Engineering
    print("\nStep 3: Running Feature Engineering Pipeline...")
    # Force re-computation (load_cached_data=False) to demonstrate the logic
    # We use a temporary split name to avoid overwriting real cache files if they exist
    df_features = fe.prepare_data(
        labels_df=df_meta_subset,
        tracking_df=df_tracking_subset,
        helmets_df=df_helmets_subset,
        load_cached_data=False,
        split="demo_train",
    )

    # Validation: Check output
    print(f"  Generated Feature DataFrame Shape: {df_features.shape}")
    assert not df_features.empty, "Feature DataFrame is empty!"
    assert "contact" in df_features.columns, "Target column 'contact' missing."

    # Check for specific feature columns (e.g., lags)
    lag_col = f"x_position_lag_0_1"  # Player 1 current x position
    assert lag_col in df_features.columns, f"Expected feature {lag_col} not found."

    # Check for visual features
    vis_col = f"{config.VISUAL_FEATURES[0]}_1"
    assert (
        vis_col in df_features.columns
    ), f"Expected visual feature {vis_col} not found."

    # 4. Dataset Creation
    print("\nStep 4: Creating PyTorch Dataset...")
    train_dataset = ds.ContactDataset(df_features, split="train")

    # Validation: Check item retrieval
    sample = train_dataset[0]
    kin_shape = sample["kinematic"].shape
    vis_shape = sample["visual"].shape
    target_shape = sample["target"].shape

    print(
        f"  Sample Shapes -> Kinematic: {kin_shape}, Visual: {vis_shape}, Target: {target_shape}"
    )

    # Assertions
    assert isinstance(sample["kinematic"], torch.Tensor)
    assert isinstance(sample["visual"], torch.Tensor)
    assert len(kin_shape) == 1, "Kinematic features should be a 1D vector per sample."
    assert len(vis_shape) == 1, "Visual features should be a 1D vector per sample."

    # 5. Model Initialization
    print("\nStep 5: Initializing NR-PIRV-Net Model...")
    kin_dim = kin_shape[0]
    vis_dim = vis_shape[0]

    net = model.NRPIRVNet(kin_input_dim=kin_dim, vis_input_dim=vis_dim)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"  Model moved to {device}")

    # 6. Training Simulation (Forward & Backward Pass)
    print("\nStep 6: Simulating Training Loop...")

    # Create DataLoader
    # Small batch size for demo
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=True
    )

    # Loss and Optimizer
    criterion = utils.FocalLoss(**config.FOCAL_LOSS_PARAMS)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    net.train()

    # Run 1 epoch (or just a few batches)
    print("  Running forward/backward passes...")
    for i, batch in enumerate(train_loader):
        kin_input = batch["kinematic"].to(device)
        vis_input = batch["visual"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # Match logit shape (N, 1)

        optimizer.zero_grad()

        # Forward
        logits = net(kin_input, vis_input)

        # Validation: Output shape
        assert (
            logits.shape == targets.shape
        ), f"Shape mismatch: Logits {logits.shape} vs Targets {targets.shape}"

        # Loss
        loss = criterion(logits, targets)

        # Backward
        loss.backward()
        optimizer.step()

        if i == 0:
            print(f"  Batch 0 Loss: {loss.item():.4f}")

        # Limit to 5 batches for speed
        if i >= 4:
            break

    print("  Training simulation completed successfully.")

    # 7. Inference & Threshold Optimization
    print("\nStep 7: Optimizing Threshold (Validation Mock)...")

    # Switch to eval mode
    net.eval()

    # Generate mock predictions on the same subset (just for demo logic)
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in train_loader:
            kin_input = batch["kinematic"].to(device)
            vis_input = batch["visual"].to(device)
            targets = batch["target"].to(device)

            logits = net(kin_input, vis_input)
            probs = torch.sigmoid(logits).squeeze()

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if len(all_targets) > 500:  # Limit size
                break

    # Optimize Threshold
    # Ensure we have mixed classes for meaningful MCC, otherwise mock it
    y_true = np.array(all_targets)
    y_scores = np.array(all_probs)

    if len(np.unique(y_true)) < 2:
        print(
            "  Warning: Subset contains only one class. Injecting dummy data for threshold demo."
        )
        y_true = np.concatenate([y_true, [0, 1]])
        y_scores = np.concatenate([y_scores, [0.1, 0.9]])

    best_thresh, best_mcc = utils.optimize_threshold(y_true, y_scores)

    print(f"  Best Threshold: {best_thresh:.4f}")
    print(f"  Best MCC Score: {best_mcc:.4f}")

    # Validation
    assert 0.0 <= best_thresh <= 1.0, "Threshold out of bounds."
    assert -1.0 <= best_mcc <= 1.0, "MCC score out of bounds."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
