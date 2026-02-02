import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

# Import from library
from library.config import Config
from library.engine import run_training, generate_submission
from library.data import get_train_val_loaders
from library.model import AMSA_DS
from library.utils import set_seed


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Column-wise Root Mean Squared Logarithmic Error.
    y_true, y_pred: (N, 2) arrays.
    """
    # Ensure non-negative
    y_pred = np.maximum(y_pred, 0)

    # log(1+x)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Squared differences
    squared_diff = (log_true - log_pred) ** 2

    # Mean per column
    mse_per_col = np.mean(squared_diff, axis=0)

    # RMSE per column
    rmsle_per_col = np.sqrt(mse_per_col)

    # Mean of RMSEs
    return np.mean(rmsle_per_col)


def main():
    # 1. Configuration Adjustments for Fast Baseline
    # We monkey-patch the Config class to ensure execution within time limits
    Config.NUM_EPOCHS = 30
    Config.PATIENCE = 5
    # Ensure we use GPU if available
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running with Device: {Config.DEVICE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # 2. Run Training
    # This saves the best model to Config.EXECUTION_DIR/best_model.pt
    # and scalers to Config.EXECUTION_DIR/
    _ = run_training(load_cached_data=True)

    # 3. Validation Assessment
    print("\nStarting Validation Assessment...")
    device = torch.device(Config.DEVICE)

    # Load validation loader
    # We need to re-get loaders to ensure we have access to the validation set
    # The scalers are already saved/fitted during run_training
    _, val_loader = get_train_val_loaders(load_cached_data=True)

    # Load Best Model
    model = AMSA_DS().to(device)
    model_path = os.path.join(Config.EXECUTION_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print("Model file not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_targets = []
    all_preds_log = []
    all_global_feats = []

    with torch.no_grad():
        for atomic_feats, global_feats, batch_indices, targets, _ in val_loader:
            atomic_feats = atomic_feats.to(device)
            global_feats_dev = global_feats.to(device)
            batch_indices = batch_indices.to(device)

            # Forward pass
            outputs = model(atomic_feats, global_feats_dev, batch_indices)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_global_feats.append(global_feats.numpy())

    # Concatenate
    y_pred_log = np.concatenate(all_preds_log, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    X_global = np.concatenate(all_global_feats, axis=0)

    # Inverse transform predictions (model outputs log(1+x))
    y_pred = np.expm1(y_pred_log)

    # Compute Metric
    metric = calculate_rmsle(y_true, y_pred)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate error magnitude (MSE on log scale per sample)
    # Error = mean((log(1+y) - pred)^2) over the 2 targets
    log_true = np.log1p(y_true)
    errors = np.mean((log_true - y_pred_log) ** 2, axis=1)

    # Create DataFrame for correlation
    # Global features: 20 dims.
    feature_names = [
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
        "density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "stoich_O",
        "n_atoms",
        "ar_ab",
        "ar_bc",
        "ar_ca",
        "w_mass",
        "w_radius",
        "w_en",
        "ang_distortion",
    ]

    # Ensure dimensions match
    if X_global.shape[1] == len(feature_names):
        df_analysis = pd.DataFrame(X_global, columns=feature_names)
        df_analysis["error"] = errors

        correlations = (
            df_analysis.corr()["error"]
            .drop("error")
            .sort_values(key=abs, ascending=False)
        )
        print("Top correlations between Input Features and Error Magnitude:")
        print(correlations.head(5))
    else:
        print(
            f"Global feature dimension {X_global.shape[1]} does not match expected {len(feature_names)}. Skipping detailed correlation."
        )

    # 5. Submission
    threshold = 0.04819517582654953
    if metric < threshold:
        print(f"\nMetric {metric} < {threshold}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"\nMetric {metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
