import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import provided library functions
from library.utils import set_seed, get_device, load_data
from library.train import train_kfold
from library.model import MSD_SE_CNN


def predict_ensemble(X, angles, checkpoint_dir, device):
    """
    Loads 5 models from checkpoints and averages their predictions.
    """
    models = []
    # Load 5 folds
    for fold_idx in range(5):
        model_path = os.path.join(checkpoint_dir, f"model_best_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold_idx} not found at {model_path}")
            continue

        model = MSD_SE_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No models loaded for inference.")

    # Inference Loop
    batch_size = 32
    num_samples = len(X)
    all_preds = []

    # Convert to tensor
    X_tensor = torch.from_numpy(X).float()
    # Ensure angles are (N, 1)
    angles_tensor = torch.from_numpy(angles).float().unsqueeze(1)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X_tensor[i : i + batch_size].to(device)
            batch_angles = angles_tensor[i : i + batch_size].to(device)

            batch_preds = []
            for model in models:
                logits = model(batch_X, batch_angles)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average across models for this batch
            batch_avg = np.mean(batch_preds, axis=0)
            all_preds.append(batch_avg)

    return np.concatenate(all_preds, axis=0)


def perform_failure_analysis(X, angles, y_true, y_pred):
    """
    Analyzes correlations between error magnitude and input features.
    """
    # Calculate errors
    y_pred_flat = y_pred.flatten()
    errors = np.abs(y_true - y_pred_flat)

    # Calculate Image Stats
    # X shape: (N, 3, 75, 75). Channels: HH (0), HV (1), Avg (2)
    b1_means = np.mean(X[:, 0, :, :], axis=(1, 2))
    b2_means = np.mean(X[:, 1, :, :], axis=(1, 2))

    stats = {
        "Incidence Angle": angles,
        "Band 1 Mean (HH)": b1_means,
        "Band 2 Mean (HV)": b2_means,
    }

    print("Correlation between Error Magnitude and Features:")
    for name, feature_vals in stats.items():
        # Check for constant values to avoid division by zero in correlation
        if len(np.unique(feature_vals)) < 2:
            print(f"  {name}: N/A (Constant value)")
            continue

        # Calculate Pearson correlation
        corr = np.corrcoef(errors, feature_vals)[0, 1]
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Configuration
    SEED = 42
    EPOCHS = 25  # Fast baseline execution
    PATIENCE = 8
    BATCH_SIZE = 32
    CACHE_DIR = "./working/idea_21"
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    THRESHOLD = 0.18120490171618245

    set_seed(SEED)
    device = get_device()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Starting Run...")

    # 2. Train Models (5-Fold CV)
    # This saves checkpoints to ./working/idea_21/checkpoints
    print("Training models...")
    train_kfold(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        seed=SEED,
        debug=False,
        cache_dir=CACHE_DIR,
    )

    # 3. Load Validation Data (Hold-out set from metadata)
    print("Loading validation data for evaluation...")
    data = load_data(cache_dir=CACHE_DIR)
    # Unpack tuple: X_train, y_train, angles_train, ids_train, X_val, y_val, angles_val, ids_val, X_test, angles_test, ids_test
    X_val = data[4]
    y_val = data[5]
    angles_val = data[6]

    # 4. Ensemble Inference on Validation Set
    print("Performing ensemble inference on validation set...")
    val_preds = predict_ensemble(X_val, angles_val, CHECKPOINT_DIR, device)

    # 5. Calculate Metric
    # Clip predictions to avoid log(0)
    val_preds_clipped = np.clip(val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_val, val_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    perform_failure_analysis(X_val, angles_val, y_val, val_preds)

    # 7. Submission
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        X_test = data[8]
        angles_test = data[9]
        ids_test = data[10]

        # Predict
        test_preds = predict_ensemble(X_test, angles_test, CHECKPOINT_DIR, device)

        # Create DataFrame
        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds.flatten()})

        # Save
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
