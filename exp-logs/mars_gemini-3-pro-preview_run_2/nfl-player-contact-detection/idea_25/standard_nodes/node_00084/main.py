import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Ensure library modules can be imported
sys.path.append("./library")

from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import get_data
from library.model import SERVN
from library.training import Trainer, optimize_threshold, predict


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load cached data to save time
    print("Loading data...")
    train_ds, val_ds, test_ds, dims, test_ids = get_data(load_cached=True)

    # FAST BASELINE OPTIMIZATION:
    # Subsample training set to 200,000 samples to ensure quick execution within time limits
    # while maintaining enough diversity for learning.
    train_size = len(train_ds)
    sample_size = min(200000, train_size)
    print(f"Subsampling training data: Using {sample_size} of {train_size} samples.")

    indices = np.random.choice(train_size, size=sample_size, replace=False)
    train_subset = Subset(train_ds, indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing SE-RVN Model...")
    model = SERVN(
        kin_input_dim=dims["kin_input_dim"],
        vis_input_dim=dims["vis_input_dim"],
        gate_input_dim=dims["gate_input_dim"],
        num_pos=dims["num_pos"],
        num_team=dims["num_team"],
    ).to(device)

    # 4. Training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    trainer = Trainer(model, train_loader, val_loader, optimizer, device)

    # Limit epochs to 5 for fast baseline execution
    print("Starting training...")
    trainer.fit(epochs=5, patience=Config.PATIENCE)

    # 5. Validation & Threshold Optimization
    print("Performing final validation...")
    # Ensure we use the best model state found during training
    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)

    val_loss, _, val_targets, val_probs = trainer.validate()

    # Optimize threshold
    best_thresh = optimize_threshold(val_targets, val_probs)

    # Calculate Final Metric
    val_preds_bin = (val_probs > best_thresh).astype(int)
    final_mcc = compute_mcc(val_targets, val_preds_bin)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Access validation features (Kinematic) to check correlation
    # val_ds.X_kin is a torch tensor, convert to numpy
    X_kin_val = val_ds.X_kin.numpy()

    correlations = []
    # Check all kinematic features
    num_feats = X_kin_val.shape[1]

    for i in range(num_feats):
        feat_col = X_kin_val[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_col)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Kinematic Features Correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"  Feature Index {idx}: {corr:.4f}")

    # 7. Submission
    TARGET_METRIC = 0.6634847318478787

    if final_mcc > TARGET_METRIC:
        print(
            f"\nMetric ({final_mcc}) > Target ({TARGET_METRIC}). Generating submission..."
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_probs = predict(model, test_loader, device)
        test_preds = (test_probs > best_thresh).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame({"contact_id": test_ids, "contact": test_preds})

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_mcc}) <= Target ({TARGET_METRIC}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
