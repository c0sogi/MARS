import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_dataloaders
from library.model import DenseContextNet
from library.train import Trainer


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Initialize config and set random seeds for reproducibility
    Config.setup()
    set_seed(Config.SEED)

    # Fast Baseline Settings
    # We limit epochs to ensure the run completes quickly (well within 2 hours).
    # The dataset size (1728 samples) is small enough that we don't need to
    # subsample the data to meet the time constraint.
    Config.EPOCHS = 30
    Config.BATCH_SIZE = 32

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing DataLoaders...")
    # load_cached_data=True ensures we use preprocessed NPZ files if available
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # =========================================================================
    # 3. Model Initialization & Training
    # =========================================================================
    print("Initializing Model and Optimizer...")
    model = DenseContextNet().to(device)

    # Loss function (Masked MCRMSE)
    criterion = MaskedMCRMSELoss().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print("Starting Training...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Run the training loop
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # =========================================================================
    # 4. Validation & Metric Calculation
    # =========================================================================
    print("Loading best model for evaluation...")
    # Load the best state dict saved during training
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Running Validation Inference...")
    tracker = MetricTracker()

    # Store predictions and targets for failure analysis
    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Inference
            outputs = model(inputs, partner_indices)

            # Update global metric tracker
            tracker.update(outputs, targets)

            # Collect for analysis
            all_val_preds.append(outputs.cpu().numpy())
            all_val_targets.append(targets.cpu().numpy())

    # Calculate and print the final metric
    final_metric = tracker.result()
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("Performing Failure Analysis...")

    # Concatenate all batches
    val_preds_arr = np.concatenate(all_val_preds, axis=0)
    val_targets_arr = np.concatenate(all_val_targets, axis=0)

    # Load validation metadata to correlate errors with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Calculate per-sample RMSE (on scored columns and positions)
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    scored_len = Config.SCORED_SEQ_LENGTH

    # Slice to scored region
    p_sliced = val_preds_arr[:, :scored_len, scored_indices]
    t_sliced = val_targets_arr[:, :scored_len, scored_indices]

    # MSE per sample: mean over length and channels
    sample_mse = np.mean((p_sliced - t_sliced) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    val_df["error_rmse"] = sample_rmse

    # Features to analyze
    analysis_features = ["signal_to_noise", "mean_reactivity"]

    # Calculate correlations
    print("Correlation between Error (RMSE) and Metadata Features:")
    for feat in analysis_features:
        if feat in val_df.columns:
            # Drop NaNs if any
            valid_data = val_df[[feat, "error_rmse"]].dropna()
            if len(valid_data) > 0:
                corr, _ = pearsonr(valid_data[feat], valid_data["error_rmse"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Insufficient data")
        else:
            print(f"  {feat}: Feature not found")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    # Threshold check
    THRESHOLD = 0.5421870350837708

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        all_test_preds = []

        with torch.no_grad():
            for inputs, partner_indices, _ in test_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)

                # Inference
                outputs = model(inputs, partner_indices)
                all_test_preds.append(outputs.cpu().numpy())

        # Concatenate predictions: (N_samples, 107, 5)
        test_preds_arr = np.concatenate(all_test_preds, axis=0)

        # Prepare submission DataFrame
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds_arr[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()

                row_dict = {"id_seqpos": row_id}
                for col_name, val in zip(target_cols, row_values):
                    row_dict[col_name] = val

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Ensure output directory exists
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
