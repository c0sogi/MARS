import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.train import train_model, validate, generate_submission
from library.model import RBFDualStreamDeepSets
from library.data import get_datasets, collate_fn


def main():
    # 1. Train the model
    # Increased epochs to 100 to allow for full convergence.
    print("Starting model training...")
    train_model(num_epochs=Config.NUM_EPOCHS)

    # 2. Load the best model and validation data for assessment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets (cached)
    # We need the validation dataset specifically.
    # get_datasets handles the scaler fitting on train and applying to val.
    _, val_dataset, _ = get_datasets(load_cached_data=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model = RBFDualStreamDeepSets().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            "Warning: No model checkpoint found. Using untrained model for validation (this will likely fail threshold)."
        )

    model.eval()

    # 3. Validation Assessment
    # The criterion used during training was MSE on log-targets.
    # The validate function returns (loss, avg_rmsle).
    criterion = torch.nn.MSELoss()
    _, final_metric = validate(model, val_loader, criterion, device)

    # Required output format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load metadata to get interpretable features for correlation analysis
    if os.path.exists(Config.VAL_METADATA_PATH):
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

        # Collect predictions and targets to compute errors
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                atomic_features = batch["atomic_features"].to(device)
                lattice_features = batch["lattice_features"].to(device)
                batch_indices = batch["batch_indices"].to(device)
                targets = batch["targets"].to(device)

                outputs = model(atomic_features, lattice_features, batch_indices)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate error magnitude per sample.
        # Since targets are log(1+y), the difference is directly related to RMSLE.
        # We use Mean Absolute Error in log space as a proxy for "how wrong" the model is.
        errors = np.mean(np.abs(all_preds - all_targets), axis=1)

        # Ensure lengths match (dataset loader might drop last batch if drop_last=True, though not here)
        if len(errors) == len(val_meta):
            val_meta["error_magnitude"] = errors

            # Select numeric columns for correlation
            numeric_cols = val_meta.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude IDs, targets, and the error itself from features list
            cols_to_exclude = [
                "id",
                "formation_energy_ev_natom",
                "bandgap_energy_ev",
                "error_magnitude",
            ]
            feature_cols = [c for c in numeric_cols if c not in cols_to_exclude]

            # Compute correlation
            correlations = (
                val_meta[feature_cols]
                .corrwith(val_meta["error_magnitude"])
                .sort_values(ascending=False, key=abs)
            )

            print("Correlation between Error Magnitude and Input Features:")
            print(correlations)
        else:
            print(
                "Warning: Mismatch between validation metadata length and prediction count. Skipping correlation analysis."
            )
    else:
        print("Validation metadata not found. Skipping failure analysis.")

    # 5. Submission
    # Threshold defined in requirements
    threshold = 0.05781995991591556

    if final_metric < threshold:
        print(
            f"\nValidation metric {final_metric} < {threshold}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(f"\nValidation metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
