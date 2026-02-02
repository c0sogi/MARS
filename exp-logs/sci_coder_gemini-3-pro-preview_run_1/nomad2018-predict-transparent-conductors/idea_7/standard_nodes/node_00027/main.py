import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# Import from the provided library
from library.config import Config
from library.dataset import MaterialDataset, collate_batch
from library.model import SIRDSModel
from library.train import run_training, generate_submission


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Initialization and Setup
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {device}")

    # 2. Train the Model
    # run_training handles dataset creation, scaling, training loop, and saving the best model checkpoint.
    # We use the full dataset (debug=False) for the best performance.
    print("Starting model training...")
    _ = run_training(debug=False)

    # 3. Load Best Model for Validation
    # The training process saves the best model to config.MODEL_PATH based on validation loss.
    # We load this specific state to ensure we evaluate the optimal model, not the last epoch's.
    print(f"Loading best model from {config.MODEL_PATH}...")
    model = SIRDSModel(config)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. Validation Assessment
    print("Evaluating on hold-out validation set...")
    val_dataset = MaterialDataset(
        metadata_path=config.VAL_CSV,
        geometry_dir=config.GEOMETRY_DIR,
        cache_path=config.VAL_CACHE,
        load_cached_data=True,
        mode="val",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True,
    )

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            atomic = batch["atomic_features"].to(device)
            global_f = batch["global_features"].to(device)
            sym = batch["symmetry"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            # Inference
            preds = model(atomic, global_f, sym, mask)

            # Store results (keep in log scale for metric calculation)
            all_preds_log.append(preds.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_global_feats.append(global_f.cpu().numpy())

    # Concatenate all batches
    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Calculate Metric: Column-wise Root Mean Squared Logarithmic Error
    # Note: The model predicts log(1+y). The targets in val_dataset are also log(1+y).
    # Therefore, RMSE on these values IS the RMSLE on the original values.
    # Metric = Mean(RMSLE_col1, RMSLE_col2)
    mse_per_col = mean_squared_error(
        all_targets_log, all_preds_log, multioutput="raw_values"
    )
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate Mean Absolute Error (in log space) per sample to quantify prediction quality
    sample_errors = np.mean(np.abs(all_preds_log - all_targets_log), axis=1)

    # Feature names corresponding to extract_global_features in library/features.py
    feature_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
    ]

    # Compute correlation between error magnitude and global features
    correlations = {}
    for i, name in enumerate(feature_names):
        feat_values = all_global_feats[:, i]
        # Avoid correlation with constant features
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(feat_values, sample_errors)[0, 1]
            correlations[name] = corr
        else:
            correlations[name] = 0.0

    print("Correlation between Error and Global Features:")
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    # Threshold condition
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, device)
    else:
        print(
            f"\nMetric {final_metric} does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
