import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.data_processing import DataProcessor
from library.train_eval import train_model, validate, compute_mcc
from library.dataset import get_dataloaders, get_test_loader
from library.loss import FocalLoss


def main():
    # 1. Configuration
    # We use the full dataset (debug=False) but limit epochs to 6 to ensure
    # the run completes well within the 2-hour limit while providing sufficient convergence.
    config = Config(debug=False, epochs=6)

    # 2. Data Processing
    processor = DataProcessor(config)

    # 3. Training
    # This handles loading data, training the model, and finding the best threshold.
    print("Starting Model Training...")
    model, best_threshold = train_model(config, processor)

    # 4. Validation & Metrics
    print("\nPerforming Final Validation...")
    # Retrieve dataloaders to get access to the validation set
    _, val_loader = get_dataloaders(config, processor)

    # Define criterion for validation function (needed for loss calculation, though we focus on MCC)
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    # Run inference on validation set
    val_loss, y_true, y_probs = validate(model, val_loader, criterion, config.DEVICE)

    # Apply best threshold
    y_pred = (y_probs > best_threshold).astype(int)

    # Compute and print Final Validation Metric
    final_mcc = compute_mcc(y_true, y_pred)
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_true - y_probs)

    # Retrieve feature matrix (Continuous features)
    # The dataset stores tensors on CPU (unless pinned), convert to numpy
    X_cont = val_loader.dataset.X_cont.numpy()

    # Reconstruct feature names to make the analysis interpretable
    # Logic matches DataProcessor._prepare_tensors
    lags = range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1)
    feature_names = []
    for k in lags:
        suffix = f"_lag_{k}"
        # P1 Kinematics
        feature_names.extend([f"{c}{suffix}_1" for c in config.TRACKING_COLS])
        # P2 Kinematics
        feature_names.extend([f"{c}{suffix}_2" for c in config.TRACKING_COLS])
        # Relative Physics
        if config.USE_LOG_DISTANCE:
            feature_names.append(f"log_dist{suffix}")
        else:
            feature_names.append(f"dist{suffix}")
        feature_names.append(f"speed_diff{suffix}")

    # Calculate correlation between each feature and the error magnitude
    correlations = []
    num_features = X_cont.shape[1]

    # Ensure we don't go out of bounds if feature names mismatch (sanity check)
    limit = min(num_features, len(feature_names))

    for i in range(limit):
        feat_vals = X_cont[:, i]
        # Avoid correlation with constant features
        if np.std(feat_vals) > 1e-9:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            correlations.append((feature_names[i], corr))
        else:
            correlations.append((feature_names[i], 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(
        "Top 10 Features correlated with Error Magnitude (Systematic Error Patterns):"
    )
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission
    THRESHOLD_SCORE = 0.62458462731896

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation MCC ({final_mcc}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Load test data
        test_loader, test_ids = get_test_loader(config, processor)

        model.eval()
        all_probs = []

        # Inference loop
        with torch.no_grad():
            for X_cont_batch, X_cat_batch in test_loader:
                X_cont_batch = X_cont_batch.to(config.DEVICE)
                X_cat_batch = X_cat_batch.to(config.DEVICE)

                logits = model(X_cont_batch, X_cat_batch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy().flatten())

        # Concatenate predictions
        y_probs_test = np.concatenate(all_probs)
        y_pred_test = (y_probs_test > best_threshold).astype(int)

        # Create submission DataFrame
        submission = pd.DataFrame({"contact_id": test_ids, "contact": y_pred_test})

        # Save
        sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path} with {len(submission)} rows.")

    else:
        print(
            f"\nValidation MCC ({final_mcc}) did not exceed threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
