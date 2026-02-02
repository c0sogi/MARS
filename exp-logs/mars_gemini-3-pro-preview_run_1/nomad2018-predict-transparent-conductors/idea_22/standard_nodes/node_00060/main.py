import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device, rmsle
from library.data import get_dataloaders
from library.model import MSNWDSModel, train_model, generate_submission
from library.engine import validate


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # Modify Config for a fast baseline execution as per requirements
    # Reducing epochs to ensure completion within the time limit while allowing convergence
    Config.NUM_EPOCHS = 30
    print(f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Data Loading]")
    # load_cached=True allows utilizing precomputed features if available in ./working
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached=True
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[Model Initialization]")
    model = MSNWDSModel()
    model.to(device)
    print("Model created.")

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print("\n[Training]")
    # train_model handles the loop, optimizer, scheduler, and early stopping
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=device,
    )

    # -------------------------------------------------------------------------
    # 5. Final Validation Assessment
    # -------------------------------------------------------------------------
    print("\n[Final Validation]")
    # Use the same loss function as training for consistency in loss reporting
    criterion = torch.nn.MSELoss()

    # validate returns loss and column-wise RMSLEs
    val_loss, val_rmsle_form, val_rmsle_band = validate(
        model, val_loader, criterion, device
    )

    # The competition metric is the mean of the column-wise RMSLEs
    final_metric = (val_rmsle_form + val_rmsle_band) / 2.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Failure Analysis]")
    model.eval()

    errors = []
    global_features_list = []

    # Collect predictions and features for correlation analysis
    with torch.no_grad():
        for batch in val_loader:
            atomic = batch["atomic"].to(device)
            glob = batch["global"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic, glob, mask)

            # Inverse transform from log1p space to original space
            preds_orig = torch.expm1(outputs).cpu().numpy()
            preds_orig = np.maximum(preds_orig, 0.0)

            targets_orig = torch.expm1(targets).cpu().numpy()
            targets_orig = np.maximum(targets_orig, 0.0)

            # Calculate Mean Absolute Error per sample across the two targets
            # Shape: (Batch_Size,)
            batch_errors = np.mean(np.abs(preds_orig - targets_orig), axis=1)

            errors.extend(batch_errors)
            global_features_list.append(glob.cpu().numpy())

    errors = np.array(errors)
    global_features_all = np.concatenate(global_features_list, axis=0)

    # Define feature names based on get_global_features in preprocessing.py
    # 0-2: Lattice Lengths (a, b, c)
    # 3-5: Lattice Angles (alpha, beta, gamma)
    # 6: Volume
    # 7: Density
    # 8-10: Stoichiometry (Al, Ga, In)
    # 11: Total Atoms
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
        "Total_Atoms",
    ]

    # Create DataFrame
    analysis_df = pd.DataFrame(global_features_all, columns=feature_names)
    analysis_df["Error_Magnitude"] = errors

    # Calculate correlation
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 Features correlated with Error Magnitude:")
    for feat, corr_val in top_correlations.items():
        # Retrieve original sign
        sign_corr = correlations[feat]
        print(f"  {feat:<15}: {sign_corr:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    print("\n[Submission Generation]")
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(f"Validation metric {final_metric} is lower than threshold {THRESHOLD}.")
        print("Generating submission file...")
        generate_submission(model, test_loader, device=device)
    else:
        print(
            f"Validation metric {final_metric} is NOT lower than threshold {THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
