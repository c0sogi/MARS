import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import set_seed, compute_rmsle, TargetScaler
from library.data import get_dataloaders
from library.model import RadiusGatedCGN
from library.train import Trainer


def main():
    # 1. Setup Configuration
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Configure for fast baseline execution while maintaining performance
    # We use the full dataset (None) because it is small (~2400 samples)
    # and we need high accuracy to pass the threshold.
    Config.DEBUG_SAMPLE_SIZE = None
    Config.NUM_EPOCHS = 100  # Sufficient for convergence on this dataset size

    # Setup directories
    Config.setup()

    # 2. Prepare Data
    # load_cached_data=True allows using pre-processed graphs if available
    print("Loading and preparing data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=True
    )

    # 3. Initialize Model
    print("Initializing RadiusGatedCGN model...")
    model = RadiusGatedCGN()

    # 4. Train Model
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, scaler, Config)
    trainer.run()

    # 5. Validation Assessment
    print("Evaluating on validation set...")
    # Load the best checkpoint saved during training
    best_model_path = Config.CHECKPOINT_PATH
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No checkpoint found. Using current model state.")

    model.eval()

    # Compute metrics on the full validation set
    val_loss, val_rmsle = trainer.evaluate(val_loader)

    # Print the final validation metric in the required format
    print(f"Final Validation Metric: {val_rmsle}")

    # 6. Failure Analysis
    print("Performing failure analysis...")

    # Collect predictions, targets, and IDs for analysis
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(Config.DEVICE)
            outputs = model(batch)

            # Inverse transform predictions to original scale
            preds_np = outputs.cpu().numpy()
            preds_original = scaler.inverse_transform(preds_np)

            # Targets are already in original scale in the dataset object,
            # but let's take them from the batch for alignment (they are tensors here)
            targets_np = batch.y.cpu().numpy()

            val_preds.append(preds_original)
            val_targets.append(targets_np)
            val_ids.append(batch.id.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_ids = np.concatenate(val_ids, axis=0)

    # Calculate error magnitude per sample (Mean Squared Logarithmic Error per sample)
    # Clip to 0 to avoid log errors
    val_preds_clipped = np.maximum(val_preds, 0)
    val_targets_clipped = np.maximum(val_targets, 0)

    log_diff = np.log1p(val_preds_clipped) - np.log1p(val_targets_clipped)
    # Mean over the two targets (formation energy and bandgap)
    error_magnitude = np.mean(log_diff**2, axis=1)

    # Create analysis DataFrame
    analysis_df = pd.DataFrame({"id": val_ids, "error_magnitude": error_magnitude})

    # Load metadata to get features
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge error with metadata
    full_analysis = pd.merge(val_metadata, analysis_df, on="id")

    # Calculate correlations between error magnitude and numerical features
    # Filter for numerical columns only
    numeric_cols = full_analysis.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude targets, id, and the error itself
    exclude_cols = [
        "id",
        "error_magnitude",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in feature_cols:
        if full_analysis[col].std() > 1e-9:  # Skip constant columns
            corr = full_analysis[col].corr(full_analysis["error_magnitude"])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 correlations between Error Magnitude and Features:")
    for feature, corr in sorted_corrs[:5]:
        print(f"  {feature}: {corr:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.049412816762924194

    if val_rmsle < THRESHOLD:
        print(
            f"Validation metric {val_rmsle} is below threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(Config.DEVICE)
                outputs = model(batch)

                # Inverse transform
                preds_np = outputs.cpu().numpy()
                preds_original = scaler.inverse_transform(preds_np)

                # Clip negative values
                preds_original = np.maximum(preds_original, 0)

                test_preds.append(preds_original)
                test_ids.append(batch.id.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID
        submission = submission.sort_values("id")

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_rmsle} is NOT below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
