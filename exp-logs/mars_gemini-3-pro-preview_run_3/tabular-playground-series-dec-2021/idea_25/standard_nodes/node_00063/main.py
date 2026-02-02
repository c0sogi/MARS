import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders, process_data
from library.train import run_training


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Execution started on device: {device}")

    # 2. Train Model
    # We limit epochs to 10 for a fast baseline execution as requested.
    # The run_training function returns the model with the best weights loaded.
    print("Starting training pipeline...")
    model = run_training(epochs=10, batch_size=Config.BATCH_SIZE, load_cached_data=True)
    model.to(device)
    model.eval()

    # 3. Validation & Metric Calculation
    print("Performing validation...")
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    all_preds = []
    all_targets = []
    all_probs = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            # Store inputs for failure analysis (move to CPU to save GPU memory)
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_probs = np.concatenate(all_probs)
    X_val = np.concatenate(all_inputs)

    # Calculate Accuracy
    accuracy = np.mean(y_pred == y_true)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {accuracy}")

    # 4. Failure Analysis
    print("\nRunning Failure Analysis...")
    # Calculate Error Magnitude: 1 - Probability of the true class
    # y_true contains indices 0-6. We grab the probability assigned to the true class.
    rows = np.arange(len(y_true))
    true_class_probs = y_probs[rows, y_true]
    error_magnitude = 1.0 - true_class_probs

    # Calculate correlation between Error Magnitude and each Feature
    # X_val is (N, Features), error_magnitude is (N,)
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = X_val[:, i]
        # Handle potential constant columns to avoid division by zero in correlation
        if np.std(feature_col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, error_magnitude)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")

    # 5. Submission
    THRESHOLD = 0.9625041666666667

    if accuracy > THRESHOLD:
        print(
            f"\nValidation metric ({accuracy}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Get Test Data
        _, _, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, load_cached_data=True
        )
        # We need test_ids. process_data returns them.
        # process_data returns: X_train, y_train, X_val, y_val, X_test, test_ids
        _, _, _, _, _, test_ids = process_data(load_cached_data=True)

        test_preds = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_preds.append(predicted.cpu().numpy())

        final_preds = np.concatenate(test_preds)

        # Map 0-6 back to 1-7
        final_preds = final_preds + 1

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({accuracy}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
