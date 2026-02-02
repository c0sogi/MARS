import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib

# Import from the provided library files
from library.config import Config
from library.dataset import prepare_dataloaders
from library.model import GRVCNet
from library.loss import FocalLoss
from library.trainer import train_one_epoch, evaluate, optimize_threshold, set_seed
from library.feature_engineering import generate_dataset


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast baseline configuration
    # We reduce epochs to ensure execution finishes well within the 2-hour limit
    FAST_EPOCHS = 5

    # 2. Data Loading
    # prepare_dataloaders handles loading, splitting, and scaling
    # We use cached data to speed up the process
    train_loader, val_loader, test_loader, dims = prepare_dataloaders(
        load_cached_data=True
    )
    kin_dim, vis_dim = dims

    # 3. Model Initialization
    model = GRVCNet(kin_dim, vis_dim, Config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # 4. Training Loop
    best_val_mcc = -1.0
    best_model_state = None

    print(f"Starting training for {FAST_EPOCHS} epochs...")
    for epoch in range(FAST_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mcc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}, Val MCC {val_mcc:.6f}"
        )

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = model.state_dict()

    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 5. Validation & Threshold Optimization
    print("Optimizing threshold on validation set...")
    _, _, val_probs, val_targets = evaluate(model, val_loader, criterion, device)
    best_thresh, final_mcc = optimize_threshold(val_targets, val_probs)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    try:
        # Calculate error magnitude
        errors = np.abs(val_targets - val_probs)

        # Retrieve feature names to make analysis meaningful
        # We load the cached dataframe to get columns, then apply the same exclusion logic as dataset.py
        df_tmp = generate_dataset(mode="train", load_cached_data=True)
        all_cols = df_tmp.columns
        vis_cols = [c for c in all_cols if c.startswith("v_")]
        exclude_cols = [
            "contact_id",
            "game_play",
            "contact",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ] + vis_cols
        kin_cols = [c for c in all_cols if c not in exclude_cols]

        # Get validation features from the dataset
        # val_loader.dataset is a ContactDataset, which exposes .features (kinematic)
        X_val_kin = val_loader.dataset.features

        # Calculate correlation between each kinematic feature and the error
        correlations = []
        for i, col_name in enumerate(kin_cols):
            if i < X_val_kin.shape[1]:
                # Compute Pearson correlation
                corr = np.corrcoef(X_val_kin[:, i], errors)[0, 1]
                if not np.isnan(corr):
                    correlations.append((col_name, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top features correlated with error magnitude:")
        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")

    except Exception as e:
        print(f"Failure analysis encountered an error: {e}")

    # 7. Conditional Submission
    THRESHOLD_SCORE = 0.6634847318478787

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation Metric {final_mcc} > {THRESHOLD_SCORE}. Generating submission..."
        )

        # Run inference on test set
        _, _, test_probs, _ = evaluate(model, test_loader, criterion, device)

        # Apply optimized threshold
        binary_preds = (test_probs > best_thresh).astype(int)

        # Prepare submission dataframe
        # We load test dataset again to ensure we have the correct contact_ids aligned with predictions
        df_test = generate_dataset(mode="test", load_cached_data=True)

        submission = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": binary_preds}
        )

        # Save submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric {final_mcc} <= {THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
