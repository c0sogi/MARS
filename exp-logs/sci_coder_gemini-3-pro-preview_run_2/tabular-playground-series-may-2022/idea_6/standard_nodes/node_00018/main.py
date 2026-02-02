import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.dataset import get_datasets
from library.model import ResFunnelGLU
from library.trainer import Trainer, set_seed


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Configure for "Fast Baseline" while maintaining performance
    # We use the full dataset (DEBUG=False) to achieve the high AUC requirement.
    # Cite solution_lesson_node_00017: Deep Residual architectures require more epochs to converge
    # than shallow networks. The previous run truncated at 20 epochs (AUC 0.9952),
    # while the implicit run showed peak performance at epoch 33 (AUC 0.9957).
    # We extend training to 50 epochs to capture this peak.
    Config.DEBUG = False
    Config.MAX_EPOCHS = 50

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading datasets...")
    # Load datasets with caching enabled to speed up subsequent runs
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True, debug=Config.DEBUG)

    # Create DataLoaders
    # Pin memory and num_workers for speed on GPU
    train_loader = DataLoader(
        train_ds,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing model...")
    model = ResFunnelGLU()

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("Starting training...")
    trainer = Trainer(model)
    best_auc = trainer.fit(
        train_loader, val_loader, epochs=Config.MAX_EPOCHS, patience=Config.PATIENCE
    )

    # --------------------------------------------------------------------------
    # 5. Validation Assessment
    # --------------------------------------------------------------------------
    # Load the best model weights for final evaluation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )

    val_loss, final_auc = trainer.validate(val_loader)
    # Print metric in required format with full precision
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    trainer.model.eval()

    all_targets = []
    all_preds = []
    all_cont_features = []

    # Collect validation predictions and features
    with torch.no_grad():
        for batch in val_loader:
            cont = batch["cont"]  # Keep on CPU for numpy conversion later
            target = batch["target"]

            # Move to device for inference
            cont_dev = cont.to(Config.DEVICE)
            cat_dev = batch["cat"].to(Config.DEVICE)

            output = trainer.model(cont_dev, cat_dev)

            all_targets.append(target.numpy())
            all_preds.append(output.cpu().numpy())
            all_cont_features.append(cont.numpy())

    all_targets = np.concatenate(all_targets).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_cont_features = np.concatenate(all_cont_features, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate Correlation with Continuous Features
    correlations = []
    # Note: Features are indexed 0-29.
    for i in range(all_cont_features.shape[1]):
        feat_vals = all_cont_features[:, i]
        # Check for zero variance to avoid division by zero in correlation
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((f"cont_feat_{i:02d}", corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 7. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9952920431395679

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                cont = batch["cont"].to(Config.DEVICE)
                cat = batch["cat"].to(Config.DEVICE)

                output = trainer.model(cont, cat)
                test_preds.append(output.cpu().numpy())

        test_preds = np.concatenate(test_preds).flatten()

        # Load Test IDs
        test_meta = pd.read_csv(Config.TEST_META_PATH)
        submission = pd.DataFrame({"id": test_meta["id"], "target": test_preds})

        save_path = os.path.join(submission_dir, "submission.csv")
        submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
