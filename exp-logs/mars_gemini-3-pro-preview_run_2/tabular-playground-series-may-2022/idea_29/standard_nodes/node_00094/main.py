import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config, set_seed, generate_submission, load_and_process_data
from library.train import run_training
from library.model import ContextAwareSwishGatedResFunnel


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    # 2. Train the model
    # We use the full dataset (data_fraction=1.0) to ensure the model learns complex interactions
    # required to beat the high threshold.
    # We limit epochs to 30. On an A100, this is extremely fast (minutes) and sufficient
    # for this architecture to converge, satisfying the "fast baseline" constraint.
    print("Starting training pipeline...")
    best_auc = run_training(data_fraction=1.0, epochs=30)

    # 3. Report Final Validation Metric
    # Requirement: Print full precision without rounding
    print(f"Final Validation Metric: {best_auc}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load validation data from cache
    # load_and_process_data returns a tuple:
    # (X_train_cont, X_train_cat, y_train, X_val_cont, X_val_cat, y_val, X_test_cont, X_test_cat, test_ids)
    data = load_and_process_data(load_cached_data=True)
    X_val_cont = data[3]
    X_val_cat = data[4]
    y_val = data[5]

    # Load the best model saved during training
    model = ContextAwareSwishGatedResFunnel().to(Config.DEVICE)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Best model not found for failure analysis.")
        return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    # Run Inference on Validation Set
    batch_size = Config.BATCH_SIZE
    preds_list = []

    # Ensure no gradients are computed for speed and memory efficiency
    with torch.no_grad():
        num_samples = len(X_val_cont)
        for i in range(0, num_samples, batch_size):
            # Prepare batch and move to device
            batch_cont = torch.FloatTensor(X_val_cont[i : i + batch_size]).to(
                Config.DEVICE
            )
            batch_cat = torch.LongTensor(X_val_cat[i : i + batch_size]).to(
                Config.DEVICE
            )

            # Forward pass
            logits = model(batch_cont, batch_cat).squeeze()
            preds = torch.sigmoid(logits)
            preds_list.append(preds.cpu().numpy())

    val_preds = np.concatenate(preds_list)

    # Calculate Error Magnitude
    errors = np.abs(y_val - val_preds)

    # Calculate Correlation with Continuous Features
    # Reconstruct feature names based on Config logic: f_00 to f_30 excluding f_27
    cont_feature_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    correlations = []
    for idx, feature_name in enumerate(cont_feature_names):
        # Extract feature column
        feature_values = X_val_cont[:, idx]
        # Compute Pearson correlation coefficient
        corr = np.corrcoef(feature_values, errors)[0, 1]
        correlations.append((feature_name, corr))

    # Sort by absolute correlation strength (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.6f}")

    # 5. Conditional Submission
    THRESHOLD = 0.9972336610045187

    if best_auc > THRESHOLD:
        print(
            f"\nValidation metric {best_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation metric {best_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
