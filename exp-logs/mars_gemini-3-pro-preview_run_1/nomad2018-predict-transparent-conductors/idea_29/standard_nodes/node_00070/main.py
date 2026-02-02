import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import configuration
from library.config import (
    VAL_CSV,
    WORKING_DIR,
    ATOMIC_HIDDEN_DIM,
    GLOBAL_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    DROPOUT,
    BATCH_SIZE,
)

# Import library functions
from library.train import train_model, set_seed
from library.predict import generate_predictions
from library.dataset import MaterialDataset, collate_materials
from library.model import PIGWDS


def main():
    # 1. Train the model
    # We use 100 epochs to ensure a good balance between speed and convergence on this dataset.
    print("Starting model training...")
    train_model(epochs=100)

    # 2. Validation and Metric Calculation
    print("\nStarting validation assessment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation using device: {device}")

    # Load the scalers generated during training
    scalers_path = os.path.join(WORKING_DIR, "scalers.npz")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(
            f"Scalers file not found at {scalers_path}. Training may have failed."
        )

    with np.load(scalers_path) as data:
        scalers = {key: data[key] for key in data.files}

    # Initialize validation dataset
    val_dataset = MaterialDataset(
        metadata_path=VAL_CSV, scalers=scalers, load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Determine input dimensions from the first sample
    sample = val_dataset[0]
    atomic_input_dim = sample["atomic_features"].shape[1]
    global_input_dim = sample["global_features"].shape[0]
    output_dim = 2  # Formation energy and Bandgap energy

    # Initialize model structure
    model = PIGWDS(
        atomic_input_dim=atomic_input_dim,
        global_input_dim=global_input_dim,
        atomic_hidden_dim=ATOMIC_HIDDEN_DIM,
        global_hidden_dim=GLOBAL_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        output_dim=output_dim,
        dropout=DROPOUT,
    ).to(device)

    # Load the best model weights saved during training
    model_path = os.path.join(WORKING_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model weights not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference loop
    preds_list = []
    targets_list = []
    global_feats_list = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(atomic_feats, global_feats, mask)

            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            global_feats_list.append(global_feats.cpu().numpy())

    # Concatenate results
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    global_features_np = np.concatenate(global_feats_list, axis=0)

    # Calculate Metric: Column-wise Root Mean Squared Logarithmic Error
    # Note: The model predicts log(1+y) and targets are log(1+y).
    # Therefore, RMSE on these values IS the RMSLE on the original scale.
    mse_per_column = np.mean((preds - targets) ** 2, axis=0)
    rmse_per_column = np.sqrt(mse_per_column)
    final_metric = np.mean(rmse_per_column)

    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate mean absolute error per sample (on log scale)
    errors = np.abs(preds - targets).mean(axis=1)

    # Feature names corresponding to the global feature vector construction in data_utils.py
    feature_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Volume",
        "Density",
        "Num_Atoms",
        "Frac_Al",
        "Frac_Ga",
        "Frac_In",
        "Avg_Mass",
        "Avg_Radius",
        "Avg_Electronegativity",
    ]

    # Compute correlation between error magnitude and each global feature
    correlations = []
    for i in range(global_features_np.shape[1]):
        feat_values = global_features_np[:, i]
        # Avoid correlation calculation if feature has zero variance
        if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            correlations.append((feature_names[i], corr))
        else:
            correlations.append((feature_names[i], 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Prediction Error and Global Features:")
    for name, corr in correlations:
        print(f"  {name:<25}: {corr:.4f}")

    # 4. Submission Generation
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} is NOT lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
