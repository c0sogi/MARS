import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.feature_engineering import FeatureEngineer
from library.dataset import DataProcessor
from library.models import SSERVN
from library.losses import FocalLoss
from library.train_eval import train_epoch, evaluate, load_validation_data


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Preparation
    fe = FeatureEngineer()
    dp = DataProcessor()

    # --- Train Data ---
    print("\nPreparing Training Data...")
    # Force regeneration to ensure full dataset usage (Cite debug_lesson_1)
    df_train = fe.generate_features(split="train", load_cached=False)

    # Capture feature names for Failure Analysis later
    # Replicating logic from DataProcessor.get_dataset to ensure alignment
    exclude_cols = {
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "team_1",
        "position_1",
        "team_2",
        "position_2",
        "path_endzone",
        "path_sideline",
        "path_all29",
        "view_1",
        "view_2",
    }
    vis_base = ["left", "top", "width", "height"]
    vis_cols = [f"{c}_1" for c in vis_base] + [f"{c}_2" for c in vis_base]
    kin_feature_names = sorted(
        [c for c in df_train.columns if c not in exclude_cols and c not in vis_cols]
    )

    train_dataset = dp.get_dataset(df_train, split="train", fit_scalers=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    del df_train
    gc.collect()

    # --- Validation Data ---
    print("\nPreparing Validation Data...")
    # Force regeneration to ensure full dataset usage (Cite debug_lesson_1)
    df_val = load_validation_data(fe, load_cached=False)
    val_dataset = dp.get_dataset(df_val, split="validation", fit_scalers=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    del df_val
    gc.collect()

    # 3. Model Initialization
    kin_dim = train_dataset.X_kin.shape[1]
    vis_dim = train_dataset.X_vis.shape[1]

    # Determine categorical cardinalities
    cat_cardinalities = []
    for i in range(4):
        max_idx = max(
            train_dataset.X_cat[:, i].max().item(), val_dataset.X_cat[:, i].max().item()
        )
        cat_cardinalities.append(max_idx + 1)

    print(f"\nModel Input Dims: Kinematic={kin_dim}, Visual={vis_dim}")
    print(f"Categorical Cardinalities: {cat_cardinalities}")

    model = SSERVN(kin_dim, vis_dim, cat_cardinalities).to(device)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    best_mcc = -1.0
    best_threshold = 0.5
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.pth")

    print(f"\nStarting Training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_probs, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        curr_thresh, curr_mcc = optimize_threshold(val_targets, val_probs, steps=100)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCC: {curr_mcc:.4f}"
        )

        if curr_mcc > best_mcc:
            best_mcc = curr_mcc
            best_threshold = curr_thresh
            torch.save(model.state_dict(), best_model_path)

    # 5. Validation Reporting
    print(f"Final Validation Metric: {best_mcc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions on validation set
    _, val_probs, val_targets = evaluate(model, val_loader, criterion, device)

    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_probs)

    # Correlate errors with Kinematic features
    # val_dataset.X_kin is a torch tensor, convert to numpy
    X_kin_val = val_dataset.X_kin.numpy()

    correlations = []
    for i, feature_name in enumerate(kin_feature_names):
        if i < X_kin_val.shape[1]:
            feat_values = X_kin_val[:, i]
            # Handle potential constant features (std=0) to avoid NaN correlation
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                correlations.append((feature_name, corr))
            else:
                correlations.append((feature_name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Model Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    print(f"\nGenerating Submission (Best MCC: {best_mcc})...")

    # Load Test Data
    # Force regeneration to ensure fresh features (Cite debug_lesson_1)
    df_test = fe.generate_features(split="test", load_cached=False)
    test_dataset = dp.get_dataset(df_test, split="test", fit_scalers=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Predict
    _, test_probs, _ = evaluate(model, test_loader, criterion, device)

    # Apply Threshold
    test_preds = (test_probs >= best_threshold).astype(int)

    # Save Submission
    submission = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": test_preds}
    )

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with {len(submission)} rows.")


if __name__ == "__main__":
    main()
