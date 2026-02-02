import os
import random
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.data import get_dataloaders
from library.train import Trainer
from library.utils import compute_rmsle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting runfile execution...")

    # 2. Data Loading
    # Load cached data if available to save time
    print("Loading data...")
    # Force reload to apply MAX_NEIGHBORS filtering
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # 3. Model Training
    print("Initializing trainer...")
    trainer = Trainer()

    # Run training
    print("Starting training loop...")
    trainer.train_loop(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # 4. Validation & Metric Calculation
    print("Loading best model for validation...")
    trainer.load_best_model()

    print("Running inference on validation set...")
    val_ids, val_preds = trainer.predict(val_loader)

    # Load ground truth from metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
    val_df = pd.read_csv(val_meta_path)

    # Create a dataframe for predictions to merge easily
    pred_df = pd.DataFrame(
        {
            "id": val_ids,
            "pred_formation": val_preds[:, 0],
            "pred_bandgap": val_preds[:, 1],
        }
    )

    # Merge on ID to ensure alignment
    merged_df = pd.merge(val_df, pred_df, on="id")

    # Extract aligned arrays
    y_true = merged_df[["formation_energy_ev_natom", "bandgap_energy_ev"]].values
    y_pred = merged_df[["pred_formation", "pred_bandgap"]].values

    # Compute Metric
    final_metric = compute_rmsle(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample (averaged over the two targets)
    # We look at magnitude of error to see what correlates with "hard" samples
    error_per_sample = np.mean(np.abs(y_true - y_pred), axis=1)
    merged_df["error_magnitude"] = error_per_sample

    # Select numerical features for correlation analysis
    feature_cols = [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    correlations = {}
    for col in feature_cols:
        if col in merged_df.columns:
            corr = merged_df[col].corr(merged_df["error_magnitude"])
            correlations[col] = corr

    # Sort and print correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Correlation between Error Magnitude and Input Features:")
    for feat, corr in sorted_corrs:
        print(f"  {feat:<30}: {corr:.4f}")

    # 6. Submission Generation
    # Threshold from requirements
    THRESHOLD = 0.053007537912991315

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on test set
        test_ids, test_preds = trainer.predict(test_loader)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID
        submission_df = submission_df.sort_values("id")

        # Save
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric ({final_metric}) is NOT below threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
