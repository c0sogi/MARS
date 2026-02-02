import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

# Import from provided libraries
from library.config import Config
from library.train import train_model
from library.data import get_dataset
from library.model import MSR_CGCNN
from library.utils import set_seed, compute_rmsle, StandardScaler, load_checkpoint


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train the model
    # We use the provided train_model function which handles data loading, scaling, and training loop.
    # It saves the best model checkpoint to Config.BEST_MODEL_PATH.
    print("Starting model training...")
    train_model(
        max_epochs=Config.MAX_EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # 3. Load the best model for evaluation
    print("Loading best model for evaluation...")
    model = MSR_CGCNN(config=Config).to(device)
    checkpoint = load_checkpoint(Config.BEST_MODEL_PATH, model, device=device)
    if checkpoint is None:
        print("Error: Could not load best model checkpoint.")
        return
    model.eval()

    # 4. Load Scalers (fit during training)
    target_scaler = StandardScaler()
    target_scaler.load_state_dict(
        torch.load(Config.TARGET_SCALER_PATH, map_location="cpu", weights_only=False)
    )

    # 5. Validation Assessment
    print("Evaluating on validation set...")
    val_dataset = get_dataset("val", load_cached_data=True, debug=Config.DEBUG)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            val_preds.append(out.cpu().numpy())
            val_targets.append(batch.y.cpu().numpy())
            val_ids.extend(batch.material_id.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Inverse transform to original scale
    val_preds_orig = target_scaler.inverse_transform(val_preds)
    val_targets_orig = target_scaler.inverse_transform(val_targets)

    # Compute Metrics
    rmsle_form = compute_rmsle(val_targets_orig[:, 0], val_preds_orig[:, 0])
    rmsle_gap = compute_rmsle(val_targets_orig[:, 1], val_preds_orig[:, 1])
    final_metric = (rmsle_form + rmsle_gap) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute errors
    abs_errors = np.abs(val_preds_orig - val_targets_orig)

    # Load metadata to get feature values for correlation
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)
    # Ensure alignment by ID
    val_results_df = pd.DataFrame(
        {
            "id": val_ids,
            "err_formation": abs_errors[:, 0],
            "err_bandgap": abs_errors[:, 1],
        }
    )
    analysis_df = val_results_df.merge(val_meta_df, on="id", how="left")

    # Calculate correlations between errors and global features
    # We use the features defined in Config
    feature_cols = Config.GLOBAL_FEATURES

    print("Correlation between Absolute Error and Features:")
    print(f"{'Feature':<30} {'Corr(Err_Form)':<20} {'Corr(Err_Gap)':<20}")
    print("-" * 75)

    for feat in feature_cols:
        if feat in analysis_df.columns:
            # Handle potential non-numeric data if any (though config features are numeric)
            if pd.api.types.is_numeric_dtype(analysis_df[feat]):
                corr_form = analysis_df["err_formation"].corr(analysis_df[feat])
                corr_gap = analysis_df["err_bandgap"].corr(analysis_df[feat])
                print(f"{feat:<30} {corr_form:<20.4f} {corr_gap:<20.4f}")

    # 7. Submission Generation
    # Threshold from instructions
    THRESHOLD = 0.05085437756413089

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = get_dataset("test", load_cached_data=True, debug=Config.DEBUG)
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                test_preds.append(out.cpu().numpy())
                test_ids.extend(batch.material_id.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Inverse transform
        test_preds_orig = target_scaler.inverse_transform(test_preds)

        # Create Submission DataFrame
        # Columns: id, formation_energy_ev_natom, bandgap_energy_ev
        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds_orig[:, 0],
                "bandgap_energy_ev": test_preds_orig[:, 1],
            }
        )

        # Sort by ID to match sample format usually
        submission_df = submission_df.sort_values("id")

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
