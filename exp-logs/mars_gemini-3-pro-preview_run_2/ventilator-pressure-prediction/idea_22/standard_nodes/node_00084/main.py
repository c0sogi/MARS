import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.config import Config
from library.utils import seed_everything, MetricMonitor
from library.data_factory import get_dataloaders
from library.model_architecture import PADIBiLSTM, WeightedL1Loss
from library.training_engine import Trainer


def run_pipeline():
    # 1. Configuration Override
    # Extended training horizon for deep convergence (Cite Lesson 00029, Lesson 00043)
    Config.EPOCHS = 180
    print(f"Configuration: Running for {Config.EPOCHS} epochs.")

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Use cached data if available to save time
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = PADIBiLSTM().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = WeightedL1Loss()

    # 5. Training
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        patience=30,  # Restore patience for long convergence
        checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    )

    trainer.fit(train_loader, val_loader, Config.EPOCHS)

    # 6. Validation Assessment & Failure Analysis
    print("\nRunning Validation Assessment...")

    # Load the best model state
    model.load_state_dict(torch.load(trainer.checkpoint_path))
    model.eval()

    val_preds = []
    val_targets = []
    val_u_out = []
    val_features = []

    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            pred = model(X)

            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_out.append(u_out.cpu().numpy())
            val_features.append(X.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds, axis=0).flatten()
    val_targets = np.concatenate(val_targets, axis=0).flatten()
    val_u_out = np.concatenate(val_u_out, axis=0).flatten()

    # Features need to be reshaped: (N_breaths, Seq_Len, Features) -> (N_total, Features)
    val_features = np.concatenate(val_features, axis=0)
    val_features = val_features.reshape(-1, val_features.shape[-1])

    # Calculate Metric: MAE on Inspiratory Phase (u_out == 0)
    insp_mask = val_u_out == 0
    final_metric = np.mean(np.abs(val_preds[insp_mask] - val_targets[insp_mask]))

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error for inspiratory phase
    errors = np.abs(val_preds - val_targets)
    insp_errors = errors[insp_mask]
    insp_features = val_features[insp_mask]

    # Reconstruct feature names
    feature_names = Config.CONTINUOUS_FEATURES + Config.BINARY_FEATURES

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(insp_features, columns=feature_names)
    df_analysis["abs_error"] = insp_errors

    # Calculate correlation with error
    correlations = df_analysis.corr()["abs_error"].sort_values(ascending=False)
    print("Correlation between Input Features and Model Error (Inspiratory Phase):")
    print(correlations.drop("abs_error"))

    # 7. Conditional Submission
    THRESHOLD = 0.1619843989610672

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                X = batch["X"].to(device)
                pred = model(X)
                test_preds.append(pred.cpu().numpy().flatten())

        all_test_preds = np.concatenate(test_preds)

        # Load metadata
        test_meta = pd.read_csv(Config.TEST_METADATA)

        if len(all_test_preds) != len(test_meta):
            print(
                f"Warning: Prediction length {len(all_test_preds)} != Metadata length {len(test_meta)}"
            )
            # Truncate if necessary (though shouldn't happen with correct loader)
            min_len = min(len(all_test_preds), len(test_meta))
            test_meta = test_meta.iloc[:min_len]
            all_test_preds = all_test_preds[:min_len]

        test_meta["pressure"] = all_test_preds

        submission = test_meta[["id", "pressure"]]
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nValidation metric {final_metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
