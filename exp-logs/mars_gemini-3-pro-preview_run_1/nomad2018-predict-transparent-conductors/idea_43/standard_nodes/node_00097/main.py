import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from library.config import Config
from library.data import get_data_loaders
from library.model import AMSP_DS_Net
from library.train_eval import Trainer, set_seed
from library.utils import inverse_log_transform


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Force regeneration to apply fixes (Cite debug_lesson_5: Reload/Regenerate)
    print("Loading data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached=False, num_workers=2
    )

    # 3. Model Initialization
    # Determine input dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    atom_dim = sample_batch["atomic_features"].shape[1]
    global_dim = sample_batch["global_features"].shape[1]

    print(f"Atomic Input Dimension: {atom_dim}")
    print(f"Global Input Dimension: {global_dim}")

    model = AMSP_DS_Net(
        atom_input_dim=atom_dim,
        global_input_dim=global_dim,
        atomic_hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        atomic_layers=Config.ATOMIC_LAYERS,
        global_hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        global_layers=Config.GLOBAL_LAYERS,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, device=device)
    trainer.fit(train_loader, val_loader)

    # 5. Validation Assessment & Failure Analysis
    print("\nPerforming Validation and Failure Analysis...")

    # Load best model
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
        print("Loaded best model checkpoint.")

    model.eval()

    val_targets_list = []
    val_preds_list = []
    val_global_feats_list = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(atomic_features, batch_indices, global_features)

            val_preds_list.append(outputs.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())
            val_global_feats_list.append(global_features.cpu().numpy())

    val_preds = np.vstack(val_preds_list)
    val_targets = np.vstack(val_targets_list)
    val_global_feats = np.vstack(val_global_feats_list)

    # Calculate Metrics (Column-wise RMSLE)
    # Since targets and preds are already log(1+x), RMSE on them is RMSLE on original
    mse = np.mean((val_preds - val_targets) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse)
    final_metric = np.mean(rmsle_per_col)

    print(f"RMSLE (Formation Energy): {rmsle_per_col[0]:.6f}")
    print(f"RMSLE (Bandgap Energy): {rmsle_per_col[1]:.6f}")
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis: Correlation between Error and Global Features
    # Error magnitude per sample per target
    errors = np.abs(val_preds - val_targets)  # Absolute error in log space

    # Global feature names (reconstructed from features.py logic for interpretability)
    # [a, b, c, alpha, beta, gamma, vol, dens, fracAl, fracGa, fracIn, N, ab, bc, ca, mass, rad, eneg]
    feat_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Volume",
        "Density",
        "Frac_Al",
        "Frac_Ga",
        "Frac_In",
        "Num_Atoms",
        "Ratio_ab",
        "Ratio_bc",
        "Ratio_ca",
        "Avg_Mass",
        "Avg_Radius",
        "Avg_Eneg",
    ]

    # Ensure we have the right number of names
    if val_global_feats.shape[1] == len(feat_names):
        print("\nCorrelation between Absolute Log-Error and Global Features:")
        print(f"{'Feature':<20} | {'Form. E. Corr':<15} | {'Bandgap Corr':<15}")
        print("-" * 55)

        for i, name in enumerate(feat_names):
            feat_vals = val_global_feats[:, i]

            # Correlation with Formation Energy Error
            corr_form, _ = pearsonr(feat_vals, errors[:, 0])

            # Correlation with Bandgap Energy Error
            corr_gap, _ = pearsonr(feat_vals, errors[:, 1])

            print(f"{name:<20} | {corr_form:15.4f} | {corr_gap:15.4f}")
    else:
        print(
            f"Skipping detailed feature correlation (Dimension mismatch: names={len(feat_names)}, feats={val_global_feats.shape[1]})"
        )

    # 6. Submission
    THRESHOLD = 0.05366557091474533

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} < Threshold {THRESHOLD}. Generating submission..."
        )

        # Predict on test set
        ids, preds = trainer.predict(test_loader)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": preds[:, 0],
                "bandgap_energy_ev": preds[:, 1],
            }
        )

        # Sort
        submission_df.sort_values("id", inplace=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
    else:
        print(
            f"\nMetric {final_metric:.6f} >= Threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
