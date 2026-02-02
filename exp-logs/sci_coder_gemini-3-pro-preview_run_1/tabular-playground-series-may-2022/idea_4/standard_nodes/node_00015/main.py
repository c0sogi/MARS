import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.data_utils import get_dataloaders, set_seeds
from library.train_utils import run_training, predict, validate
from library.model import HybridTransformer


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Adjust Config for A100 GPU and fast baseline execution
    Config.BATCH_SIZE = 2048  # Increase batch size for A100
    # Cite solution_lesson_node_00013: Tuning Learning Rate Decay via Epoch Scaling
    Config.EPOCHS = 21

    # Ensure reproducibility
    set_seeds(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading Data...")
    # load_cached_data is handled inside get_dataloaders based on Config.LOAD_CACHED_DATA
    train_loader, val_loader, test_loader = get_dataloaders()

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("\nStarting Training...")
    # run_training handles the loop, checkpointing, and returns best val AUC seen during training
    _ = run_training(train_loader, val_loader, epochs=Config.EPOCHS, patience=5)

    # -------------------------------------------------------------------------
    # 4. Final Evaluation & Metric
    # -------------------------------------------------------------------------
    print("\nPerforming Final Evaluation...")

    # Load the best model for evaluation
    model = HybridTransformer().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model checkpoint not found!")
        return

    model.eval()

    # Run validation manually to get predictions and targets for analysis
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_seq, x_num, targets in val_loader:
            x_seq = x_seq.to(device)
            x_num = x_num.to(device)

            logits = model(x_seq, x_num)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()

    final_auc = roc_auc_score(y_true, y_pred)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_auc}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude
    errors = np.abs(y_true - y_pred)

    # Get Validation Numerical Features
    # The dataset stores them as a tensor. We can access the full tensor directly.
    # Note: val_loader.dataset.numerical is a torch tensor
    X_val_num = val_loader.dataset.numerical.numpy()

    # Calculate correlation between Error and each Numerical Feature
    correlations = []
    feature_names = Config.NUM_FEATURES

    for i, feature_name in enumerate(feature_names):
        feat_values = X_val_num[:, i]
        # Calculate Pearson correlation
        if np.std(feat_values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((feature_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.6f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9920540777100928

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict(test_loader)
    else:
        print(
            f"\nValidation AUC ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
