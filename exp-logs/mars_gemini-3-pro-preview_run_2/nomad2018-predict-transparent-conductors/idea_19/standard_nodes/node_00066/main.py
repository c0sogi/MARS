import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import SR_CGN_DP
from library.train import run_training, generate_submission


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    """
    # Clip predictions to be non-negative as log is undefined for negative values
    # Formation energy and bandgap should physically be non-negative or close to it in this context
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate RMSLE for each target column separately
    rmsle_formation = np.sqrt(mean_squared_log_error(y_true[:, 0], y_pred[:, 0]))
    rmsle_bandgap = np.sqrt(mean_squared_log_error(y_true[:, 1], y_pred[:, 1]))

    # Return the average
    return (rmsle_formation + rmsle_bandgap) / 2


def main():
    # 1. Setup System
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Run Training Pipeline (Fast Baseline)
    # We limit epochs to 30 to ensure quick execution while allowing some convergence.
    print("Starting model training...")
    run_training(num_epochs=30, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Prepare Data for Validation Assessment
    # We reload dataloaders to get the validation set and the scaler used during training
    print("Loading validation data...")
    _, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 4. Load the Best Saved Model
    print("Loading best model checkpoint...")
    model = SR_CGN_DP(
        node_dim=Config.ATOM_EMBEDDING_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        rbf_bins=Config.RBF_BINS,
        rbf_min=Config.RBF_MIN,
        rbf_max=Config.RBF_MAX,
    ).to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Checkpoint not found. Using current model weights.")

    # 5. Validation Inference
    model.eval()
    val_preds = []
    val_targets = []
    val_ids = []

    print("Performing inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)

            # Inverse transform predictions and targets to original scale (eV)
            pred_np = scaler.inverse_transform(outputs.cpu().numpy())
            target_np = scaler.inverse_transform(batch.y.cpu().numpy())

            val_preds.append(pred_np)
            val_targets.append(target_np)
            val_ids.extend(batch.id.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # 6. Calculate and Print Metric
    metric_score = calculate_rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {metric_score}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample (averaged over the two targets)
    # This represents the "Error Magnitude"
    sample_errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Load metadata to correlate errors with features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Map errors to the dataframe
    # We assume the order is preserved, but using a map is safer
    error_map = dict(zip(val_ids, sample_errors))
    val_df["error_magnitude"] = val_df["id"].map(error_map)

    # Identify numerical features for correlation analysis
    exclude_cols = [
        "id",
        "file_path",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
        "error_magnitude",
    ]
    feature_cols = [
        c
        for c in val_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(val_df[c])
    ]

    correlations = {}
    for col in feature_cols:
        # Drop NaNs if any (though metadata analysis showed none)
        valid_data = val_df[[col, "error_magnitude"]].dropna()
        if len(valid_data) > 1 and valid_data[col].std() > 0:
            corr, _ = pearsonr(valid_data[col], valid_data["error_magnitude"])
            correlations[col] = corr

    # Sort and print top 5 correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top correlations between Error Magnitude and Input Features:")
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 8. Conditional Submission Generation
    # Threshold defined in requirements
    THRESHOLD = 0.049412816762924194
    submission_path = Config.SUBMISSION_PATH

    if metric_score < THRESHOLD:
        print(
            f"\nValidation metric ({metric_score}) meets the threshold ({THRESHOLD})."
        )
        print("Ensuring submission file exists...")
        # run_training generated it, but we regenerate to be explicit and ensure it uses the best model state
        generate_submission(model, test_loader, scaler, device, submission_path)
    else:
        print(
            f"\nValidation metric ({metric_score}) does NOT meet the threshold ({THRESHOLD})."
        )
        if os.path.exists(submission_path):
            print("Removing generated submission file...")
            os.remove(submission_path)
        else:
            print("No submission file to remove.")


if __name__ == "__main__":
    main()
