import os
import torch
import numpy as np
import pandas as pd
from library.config import VAL_CSV, CHECKPOINT_DIR, CACHE_DIR, DEVICE, SEED, TARGET_COLS
from library.train import run_training
from library.predict import generate_predictions
from library.data import get_dataloaders
from library.model import CrystalGraphConvNet
from library.utils import rmsle, CompositionScaler, LogStandardScaler


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Set seed for reproducibility
    set_seed(SEED)

    # 1. Train the model
    # We use 80 epochs to ensure fast execution while allowing convergence.
    # The architecture is efficient, and the dataset is small (~2k samples).
    print("Starting model training...")
    run_training(num_epochs=80, load_cached_data=True)

    # 2. Validation Assessment & Failure Analysis
    print("Starting validation assessment and failure analysis...")

    # Load DataLoaders
    # We call get_dataloaders to ensure we get the exact same split and processing as training.
    # This will also reload/refit scalers, ensuring consistency.
    # We discard train/test loaders here as we only need val_loader for analysis.
    _, val_loader, _, comp_scaler, target_scaler = get_dataloaders(
        load_cached_data=True
    )

    # Load the best model saved during training
    model = CrystalGraphConvNet().to(DEVICE)
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Run inference on validation set
    all_preds = []
    all_targets = []
    all_ids = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)

            # Forward pass
            preds_scaled = model(batch)

            # Inverse transform predictions to original scale (eV)
            preds_original = target_scaler.inverse_transform(preds_scaled)

            # Store results
            all_preds.append(preds_original.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

            # Handle IDs (can be tensor or list depending on loader collation)
            if isinstance(batch.id, torch.Tensor):
                all_ids.extend(batch.id.cpu().numpy().tolist())
            else:
                all_ids.extend(batch.id)

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Ensure non-negative predictions for Log calculation (physics constraint)
    all_preds = np.maximum(all_preds, 0)
    all_targets = np.maximum(all_targets, 0)

    # Compute Final Validation Metric
    # Using the provided utility function which implements Column-wise RMSLE
    val_metric = rmsle(all_targets, all_preds)
    print(f"Final Validation Metric: {val_metric}")

    # Failure Analysis
    # Calculate error magnitude per sample (Euclidean distance in log space)
    # This represents how far off the prediction is in the metric space
    log_pred = np.log1p(all_preds)
    log_true = np.log1p(all_targets)
    # Mean Squared Error per sample across targets, then sqrt
    error_magnitude = np.sqrt(np.mean((log_pred - log_true) ** 2, axis=1))

    # Map errors to metadata
    val_df = pd.read_csv(VAL_CSV)

    # Map error back to dataframe using ID
    id_to_error = dict(zip(all_ids, error_magnitude))
    val_df["error"] = val_df["id"].map(id_to_error)

    # Select numeric features for correlation analysis
    # Exclude IDs, paths, and target columns
    exclude_cols = ["id", "file_path", "error"] + TARGET_COLS
    feature_cols = [c for c in val_df.columns if c not in exclude_cols]
    numeric_cols = (
        val_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    )

    if numeric_cols:
        correlations = (
            val_df[numeric_cols].corrwith(val_df["error"]).sort_values(ascending=False)
        )
        print("\nCorrelation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("\nNo numeric features found for correlation analysis.")

    # 3. Generate Submission
    # Check against the threshold defined in the task
    THRESHOLD = 0.05085437756413089

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {val_metric} is NOT better than threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
