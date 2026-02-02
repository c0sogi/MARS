import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

# Import from provided libraries
from library.config import Config
from library.utils import set_seed
from library.data_processing import generate_contact_features, PIRVDataset
from library.model import PIRVNet
from library.train import Trainer


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Train Model (Fast Baseline)
    # We use a subset of data and fewer epochs to ensure execution finishes quickly
    print("\n=== Starting Training Phase ===")
    trainer = Trainer(device=device)
    trainer.fit(epochs=4, batch_size=Config.BATCH_SIZE, debug_sample=150000)

    # 3. Full Validation Inference
    print("\n=== Starting Full Validation Phase ===")

    # Load FULL validation set (not subsampled)
    val_meta_path = os.path.join(Config.METADATA_DIR, "validation.csv")
    train_tracking_path = os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
    train_helmets_path = os.path.join(Config.INPUT_DIR, "train_baseline_helmets.csv")

    X_kin_val, X_vis_val, y_val = generate_contact_features(
        val_meta_path,
        train_tracking_path,
        train_helmets_path,
        mode="val",
        load_cached_data=True,
    )

    # Load Best Model
    input_dim_kin = X_kin_val.shape[1]
    input_dim_vis = X_vis_val.shape[1]
    model = PIRVNet(input_dim_kin, input_dim_vis).to(device)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Error: Best model not found. Using current model state.")

    # Load Best Threshold
    thresh_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        best_threshold = float(np.load(thresh_path))
        print(f"Loaded best threshold: {best_threshold}")
    else:
        best_threshold = 0.5
        print("Warning: Best threshold not found. Defaulting to 0.5.")

    # Inference Loop
    val_dataset = PIRVDataset(X_kin_val, X_vis_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    all_probs = []

    with torch.no_grad():
        for x_kin, x_vis, _ in val_loader:
            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)
            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    y_pred_prob = np.vstack(all_probs).flatten()
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # Calculate Metric
    final_mcc = matthews_corrcoef(y_val, y_pred)
    print(f"Final Validation Metric: {final_mcc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - y_pred_prob)

    # We correlate error with kinematic features to find weak points
    # X_kin_val is a numpy array. We'll check the first N features (current state)
    # The features are standardized, but correlation is scale invariant.

    # Just taking a subset of features to print (e.g., current timestep features)
    # Assuming the structure from Config: [x, y, speed, accel, dir, orient, sa] for p1, p2, then dist, rels
    # We have lags. Let's just correlate with all columns and pick top 5.

    # Compute correlation vector efficiently
    # (n_samples, n_features)
    # Center errors
    e_centered = errors - errors.mean()
    # Center features
    X_centered = X_kin_val - X_kin_val.mean(axis=0)

    # Covariance
    covariance = np.dot(e_centered, X_centered) / (len(errors) - 1)
    # Std devs
    e_std = errors.std()
    X_std = X_kin_val.std(axis=0)

    correlations = covariance / (e_std * X_std + 1e-9)

    # Get top 5 absolute correlations
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top 5 Features correlated with Error Magnitude:")
    for idx in top_indices:
        print(f"Feature Index {idx}: Correlation {correlations[idx]:.4f}")

    # 5. Submission
    TARGET_METRIC = 0.6634847318478787

    if final_mcc > TARGET_METRIC:
        print("\n=== Generating Submission ===")

        # Paths for test data
        test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_tracking_path = os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
        test_helmets_path = os.path.join(Config.INPUT_DIR, "test_baseline_helmets.csv")

        # Clear memory
        del X_kin_val, X_vis_val, y_val, val_loader, val_dataset
        import gc

        gc.collect()

        # Generate Test Features
        print("Processing test features...")
        X_kin_test, X_vis_test, _ = generate_contact_features(
            test_meta_path,
            test_tracking_path,
            test_helmets_path,
            mode="test",
            load_cached_data=True,
        )

        # Test Inference
        test_dataset = PIRVDataset(X_kin_test, X_vis_test, None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_probs = []
        with torch.no_grad():
            for x_kin, x_vis in test_loader:
                x_kin = x_kin.to(device)
                x_vis = x_vis.to(device)
                logits = model(x_kin, x_vis)
                probs = torch.sigmoid(logits)
                test_probs.append(probs.cpu().numpy())

        flat_test_probs = np.vstack(test_probs).flatten()
        test_preds = (flat_test_probs >= best_threshold).astype(int)

        # Load sample submission to ensure order
        df_test = pd.read_csv(test_meta_path)
        df_sub = pd.DataFrame(
            {"contact_id": df_test["contact_id"], "contact": test_preds}
        )

        out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path} with {len(df_sub)} rows.")

    else:
        print(
            f"\nMetric {final_mcc} did not meet threshold {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    run()
