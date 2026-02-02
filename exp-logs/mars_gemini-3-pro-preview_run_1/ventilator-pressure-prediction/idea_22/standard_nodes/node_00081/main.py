import sys
import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.train import train_model
from library.dataset import get_data_loaders


def main():
    # 1. Setup
    # Ensure reproducibility across runs
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Train Model
    # Execute the training pipeline. The configuration (35 epochs) is optimized
    # for a quick yet effective baseline on this sequence-grouped dataset.
    print("Starting training process...")
    trainer = train_model(
        load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
    )

    # 3. Load Best Model
    # Ensure we use the best checkpoint saved during training, not just the last epoch.
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        )
    else:
        print("Warning: Model checkpoint not found. Using current model state.")

    trainer.model.eval()

    # 4. Validation & Failure Analysis
    print("Running validation inference...")
    # Retrieve data loaders. Since caching is enabled, this is efficient.
    _, val_loader, test_loader = get_data_loaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    val_preds = []
    val_targets = []
    val_u_outs = []
    val_inputs = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            x = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            # Forward pass (returns final_pred, aux_pred)
            pred, _ = trainer.model(x)

            # Move to CPU to conserve GPU memory for large datasets
            val_preds.append(pred.squeeze(-1).cpu())
            val_targets.append(y.cpu())
            val_u_outs.append(u_out.cpu())
            val_inputs.append(x.cpu())

    # Concatenate batches
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_u_outs = torch.cat(val_u_outs)
    val_inputs = torch.cat(val_inputs)

    # Compute Final Metric
    # Metric is MAE strictly on the inspiratory phase (u_out == 0)
    final_metric = compute_metric(val_preds, val_targets, val_u_outs)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("Performing failure analysis...")
    # Calculate absolute error per time step
    abs_errors = torch.abs(val_preds - val_targets)

    # Create mask for inspiratory phase (u_out == 0)
    mask = val_u_outs == 0

    # Filter data for analysis
    errors_masked = abs_errors[mask].numpy()
    inputs_masked = val_inputs[mask].numpy()

    # Reconstruct Feature Names based on library.features logic
    feature_names = ["time_step", "u_in", "R", "C", "volume", "R_u_in", "vol_C"]
    feature_names += [f"u_in_lag{i}" for i in range(1, Config.LAG_STEPS + 1)]
    feature_names += ["u_in_diff1", "u_in_diff2"]

    # Safety check for feature dimension
    if inputs_masked.shape[1] != len(feature_names):
        feature_names = [f"feat_{i}" for i in range(inputs_masked.shape[1])]

    # Create DataFrame
    analysis_df = pd.DataFrame(inputs_masked, columns=feature_names)
    analysis_df["abs_error"] = errors_masked

    # Compute Correlation
    corrs = (
        analysis_df.corr()["abs_error"].drop("abs_error").sort_values(ascending=False)
    )
    print("Correlation of features with Absolute Error (Inspiratory Phase):")
    print(corrs)

    # 5. Submission
    THRESHOLD = 0.2164510190486908

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        test_ids_list = []
        test_preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["X"].to(device)
                ids = batch["id"].to(device)

                # Predict
                pred, _ = trainer.model(x)

                # Flatten predictions and IDs for CSV format
                # pred shape: (B, 80, 1) -> squeeze -> (B, 80) -> flatten
                pred_flat = pred.squeeze(-1).flatten().cpu().numpy()
                ids_flat = ids.flatten().cpu().numpy()

                test_preds_list.append(pred_flat)
                test_ids_list.append(ids_flat)

        # Concatenate all test batches
        all_preds = np.concatenate(test_preds_list)
        all_ids = np.concatenate(test_ids_list)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": all_ids, "pressure": all_preds})

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission generation skipped.")


if __name__ == "__main__":
    main()
