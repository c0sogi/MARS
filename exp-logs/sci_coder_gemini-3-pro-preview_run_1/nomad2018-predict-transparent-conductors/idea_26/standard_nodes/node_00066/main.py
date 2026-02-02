import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from library import config, data, model, engine


def calculate_column_wise_rmsle(preds_log, targets_log):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.
    Since predictions and targets are already in log1p scale, this is just RMSE on them.
    """
    mse = np.mean((preds_log - targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse)
    return np.mean(rmsle_per_col)


def main():
    # 1. Setup and Reproducibility
    engine.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("Initializing Data Processor...")
    processor = data.DataProcessor()

    # Load data (this handles feature extraction, caching, and scaling)
    # Using load_cached_data=True to use pre-computed features if available
    train_loader, val_loader, test_loader = processor.process_and_get_loaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing IDCR-WDS Model...")
    net = model.IDCR_WDS_Model()

    # 4. Training
    print("Starting Training...")
    trainer = engine.Trainer(net)
    trainer.train(train_loader, val_loader)

    # 5. Load Best Model for Evaluation
    print(f"Loading best model from {config.MODEL_PATH}...")
    net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    net.to(device)
    net.eval()

    # 6. Validation Assessment & Failure Analysis
    print("Performing Validation Assessment...")
    val_preds_log = []
    val_targets_log = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atoms = batch[0].to(device)
            batch_indices = batch[1].to(device)
            glob_feats = batch[2].to(device)
            targets = batch[3].to(device)

            outputs = net(atoms, batch_indices, glob_feats)

            val_preds_log.append(outputs.cpu().numpy())
            val_targets_log.append(targets.cpu().numpy())
            val_global_feats.append(glob_feats.cpu().numpy())

    val_preds_log = np.concatenate(val_preds_log, axis=0)
    val_targets_log = np.concatenate(val_targets_log, axis=0)
    val_global_feats = np.concatenate(val_global_feats, axis=0)

    # Calculate Metric
    final_metric = calculate_column_wise_rmsle(val_preds_log, val_targets_log)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Calculate mean absolute error per sample (average over the two targets)
    errors = np.mean(np.abs(val_preds_log - val_targets_log), axis=1)

    # Feature names based on config.GLOBAL_INPUT_DIM structure
    # 0-2: Lattice Lengths, 3-5: Lattice Angles, 6: Volume, 7: Density, 8-10: Stoich, 11: N_atoms
    # Note: These are scaled values, but correlation is invariant to linear scaling.
    feature_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Volume",
        "Density",
        "Stoich_Al",
        "Stoich_Ga",
        "Stoich_In",
        "Num_Atoms",
    ]

    for i, name in enumerate(feature_names):
        if i < val_global_feats.shape[1]:
            feat_values = val_global_feats[:, i]
            # Handle potential constant values (e.g. if all angles are 90) which give NaN correlation
            if np.std(feat_values) > 1e-6:
                corr, _ = pearsonr(errors, feat_values)
                print(f"  Error vs {name:<12}: {corr:.4f}")
            else:
                print(f"  Error vs {name:<12}: N/A (Constant)")

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        engine.generate_submission(net, test_loader, device, config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
