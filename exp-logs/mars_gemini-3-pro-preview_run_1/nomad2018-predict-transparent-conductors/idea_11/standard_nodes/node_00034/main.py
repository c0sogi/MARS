import os
import torch
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
)
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import EWADeepSets
from library.train import Trainer


def main():
    # 1. Set seed for reproducibility
    set_seed(SEED)

    # 2. Load Data
    # Use cached data to speed up loading if available
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    print("Initializing model...")
    model = EWADeepSets()

    # 4. Train Model
    # Using 50 epochs for a fast baseline execution as per requirements
    print("Starting training...")
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader, epochs=50, patience=10)

    # 5. Validation & Metric Calculation
    print("Performing validation...")
    # Load best model weights
    model_path = os.path.join(WORKING_DIR, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded best model from {model_path}")

    model.eval()
    val_preds_log = []
    val_targets_log = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats = batch["atomic_features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            global_feats = batch["global_features"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            outputs = model(atomic_feats, mask, global_feats)

            val_preds_log.append(outputs.cpu().numpy())
            val_targets_log.append(targets.cpu().numpy())
            val_global_feats.append(global_feats.cpu().numpy())

    val_preds_log = np.concatenate(val_preds_log, axis=0)
    val_targets_log = np.concatenate(val_targets_log, axis=0)
    val_global_feats = np.concatenate(val_global_feats, axis=0)

    # Calculate Column-wise RMSLE
    # Note: Targets and Model Outputs are already log(1+x) transformed.
    # RMSLE = sqrt(mean((log(1+y) - log(1+pred))^2))
    # So we just calculate RMSE on the log-transformed values.
    mse_per_col = np.mean((val_preds_log - val_targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Calculate error magnitude per sample (mean absolute error in log space)
    errors = np.mean(np.abs(val_preds_log - val_targets_log), axis=1)

    # Feature names corresponding to the global_features vector constructed in library/data.py
    feat_names = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "volume",
        "density",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "percent_atom_o",
    ]

    # Calculate correlations between error and each global feature
    correlations = []
    # Ensure we don't go out of bounds if feature dim mismatches (though it shouldn't)
    num_feats = min(len(feat_names), val_global_feats.shape[1])

    for i in range(num_feats):
        feat_vals = val_global_feats[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feat_vals) > 1e-9:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
            correlations.append((feat_names[i], corr))
        else:
            correlations.append((feat_names[i], 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<30} | {'Correlation':<10}")
    print("-" * 45)
    for name, corr in correlations:
        print(f"{name:<30} | {corr:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Use trainer's predict method which handles inverse transformation
        ids, preds_original = trainer.predict(test_loader)

        submission_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": preds_original[:, 0],
                "bandgap_energy_ev": preds_original[:, 1],
            }
        )

        # Sort by ID to ensure correct order
        submission_df = submission_df.sort_values("id")

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(submission_df.head())
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
