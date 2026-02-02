import os
import sys
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_dataloaders
from library.model import DCNV2
from library.train import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Overrides
    # --------------------------------------------------------------------------
    # Optimize for A100 GPU and ensure fast execution within time limits
    Config.BATCH_SIZE = 4096
    Config.EPOCHS = 20  # Reduced from 30 to ensure timely completion

    print(
        f"Initializing pipeline with Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}"
    )
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    # Load full dataset using cached numpy arrays if available
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Determine input dimensions dynamically
    sample_cont, sample_cat, _ = next(iter(train_loader))
    num_cont_features = sample_cont.shape[1]
    print(f"Detected {num_cont_features} continuous features.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)
    model = DCNV2(num_cont_features=num_cont_features)
    model.to(device)

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_ids=test_ids,
        device=device,
    )

    # Execute training loop with early stopping
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\nRunning Validation and Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_cont_inputs = []

    # Perform inference on validation set
    # We accumulate results on CPU to avoid GPU OOM during analysis
    with torch.no_grad():
        for x_cont, x_cat, target in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            target = target.to(device)

            outputs = model(x_cont, x_cat)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_cont_inputs.append(x_cont.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_cont_inputs = np.concatenate(all_cont_inputs, axis=0)

    # Calculate and Print Final Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy:.16f}")

    # --- Failure Analysis ---
    # Calculate error vector (1 if prediction is wrong, 0 if correct)
    errors = (all_preds != all_targets).astype(int)

    # Reconstruct feature names for meaningful reporting
    try:
        # Read metadata header to get original column names
        df_cols = pd.read_parquet(Config.TRAIN_DATA_PATH).columns.tolist()

        # Identify original continuous columns
        soil_cols = [c for c in df_cols if c.startswith(Config.SOIL_PREFIX)]
        wild_cols = [c for c in df_cols if c.startswith(Config.WILDERNESS_PREFIX)]
        exclude_cols = set(soil_cols + wild_cols + [Config.ID_COL, Config.TARGET_COL])
        cont_cols = [c for c in df_cols if c not in exclude_cols]

        # Append engineered feature names (order must match data_processing.py)
        cont_cols.append("Dist_Hydro_Euclidean")
        cont_cols.append("Mean_Dist_Amenities")
        cont_cols.append("Elev_Minus_Vert_Hydro")
        cont_cols.append("Elev_Plus_Vert_Hydro")

        feature_names = cont_cols
    except Exception as e:
        print(f"Warning: Could not reconstruct feature names ({e}). Using indices.")
        feature_names = [f"Feature_{i}" for i in range(num_cont_features)]

    # Validate name count
    if len(feature_names) != all_cont_inputs.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(all_cont_inputs.shape[1])]

    # Compute correlation between each feature and the error magnitude
    print("\nFailure Analysis - Correlation with Error Magnitude:")
    correlations = []
    for i in range(all_cont_inputs.shape[1]):
        feat_vals = all_cont_inputs[:, i]
        # Handle potential constant features (avoid division by zero)
        if np.std(feat_vals) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations.append((feature_names[i], corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<30} {'Correlation':<10}")
    print("-" * 45)
    for name, corr in correlations[:10]:
        print(f"{name[:29]:<30} {corr:.6f}")

    # --------------------------------------------------------------------------
    # 6. Submission Logic
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9622416666666667

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy:.6f} > {THRESHOLD:.6f}. Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation accuracy {accuracy:.6f} <= {THRESHOLD:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
