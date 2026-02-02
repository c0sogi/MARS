import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import prepare_data
from library.model import UnifiedDeepBiLSTM
from library.train import run_training
from library.utils import seed_everything


def main():
    # 1. Configuration & Setup
    config = Config()
    # Removed epoch override to allow full convergence (Cite Lesson 00005)

    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Training
    # Removed breath limit to train on full dataset for maximum performance
    print("\n=== Starting Training ===")
    run_training(config, debug=False, limit_breaths=None)

    # 3. Full Validation
    # We must evaluate on the ENTIRE validation set.
    # We set load_cached_data=False to ignore the limited 'val_data.npz' created during training
    # and force processing of the full validation metadata.
    # It will use the 'scaler_params.npz' generated during training, which is correct.
    print("\n=== Starting Full Validation ===")
    val_loader = prepare_data("val", config, load_cached_data=False, limit_breaths=None)

    # Load Model
    # Determine input dim from a batch
    sample_batch = next(iter(val_loader))
    input_dim = sample_batch["X"].shape[-1]

    model = UnifiedDeepBiLSTM(input_dim, config).to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference Loop
    all_preds = []
    all_targets = []
    all_u_out = []
    all_features = []

    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            pred = model(X)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_features.append(X.cpu().numpy())

    # Concatenate
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    u_out_flat = np.concatenate(all_u_out).flatten()
    features_flat = np.concatenate(all_features)  # Shape: (Total_Steps, Num_Features)
    features_flat = features_flat.reshape(-1, features_flat.shape[-1])

    # Calculate Metric (Inspiratory Phase Only: u_out == 0)
    mask = u_out_flat == 0
    mae = np.mean(np.abs(preds_flat[mask] - targets_flat[mask]))

    print(f"Final Validation Metric: {mae}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Filter for inspiratory phase
    errors = np.abs(preds_flat[mask] - targets_flat[mask])
    feats_masked = features_flat[mask]

    # Feature names corresponding to library.dataset.add_features + u_out
    feature_names = [
        "time_step",
        "u_in",
        "u_in_cumsum",
        "R",
        "C",
        "R_u_in",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "dt",
        "u_out",
    ]

    # Create DataFrame for correlation analysis
    analysis_df = pd.DataFrame(feats_masked, columns=feature_names)
    analysis_df["error"] = errors

    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # 5. Submission
    threshold = 0.20567339658737183
    if mae < threshold:
        print(f"\nMetric {mae} < {threshold}. Generating submission...")

        # Load Test Data (Use cache if available, or process)
        test_loader = prepare_data("test", config, load_cached_data=True)

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                X = batch["X"].to(device)
                pred = model(X)
                test_preds.append(pred.cpu().numpy().flatten())

        test_preds = np.concatenate(test_preds)

        # Load Metadata for IDs
        test_meta = pd.read_csv(config.TEST_META)

        # Sanity check
        if len(test_preds) != len(test_meta):
            print(
                f"Warning: Prediction length {len(test_preds)} matches metadata {len(test_meta)}?"
            )

        submission = pd.DataFrame({"id": test_meta["id"], "pressure": test_preds})

        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric {mae} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
