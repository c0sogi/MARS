import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

# Import from library
from library.config import Config
from library.data_loader import get_loaders
from library.model import DC3_WDS
from library.trainer import set_seed, train_one_epoch, evaluate
from library.inference import predict, inverse_transform, generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust Config for fast baseline execution
    Config.EPOCHS = 100  # Reduced from 200 for speed

    print(f"Running on device: {device}")
    print(f"Training for {Config.EPOCHS} epochs.")

    # 2. Load Data
    # load_cached_data=True to use the pre-processed npz files in working/cache
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Initialize Model
    model = DC3_WDS().to(device)

    # 4. Optimizer and Loss
    # MSE on log-transformed targets
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=False,  # Suppress verbose output as requested
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    # We will track the model state with best validation loss
    best_model_state = None

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            # Save best model to disk as required by inference/config
            torch.save(best_model_state, Config.MODEL_PATH)

    print(f"Training finished. Best Val Loss (MSE): {best_val_loss}")

    # 6. Validation Assessment & Metric Calculation
    # Load best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    val_targets = []
    val_preds = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats = batch["atomic_feats"].to(device)
            global_feats = batch["global_feats"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(atomic_feats, global_feats, mask)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_global_feats.append(global_feats.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_global_feats = np.concatenate(val_global_feats, axis=0)

    # Metric: Column-wise Root Mean Squared Logarithmic Error
    # Since targets and preds are already log1p transformed:
    # RMSLE_col = sqrt(mean((pred - target)^2))
    mse_col = np.mean((val_preds - val_targets) ** 2, axis=0)
    rmsle_col = np.sqrt(mse_col)
    final_metric = np.mean(rmsle_col)

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Calculate error magnitude per sample (mean absolute error across targets)
    # Error in log space
    errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Global feature names (from Config/Data Loader logic)
    # [a, b, c, alpha, beta, gamma, vol, dens, al, ga, in, n_atoms]
    feature_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
        "volume",
        "density",
        "pct_al",
        "pct_ga",
        "pct_in",
        "num_atoms",
    ]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Create a dataframe for correlation
    analysis_data = pd.DataFrame(val_global_feats, columns=feature_names)
    analysis_data["error"] = errors

    correlations = (
        analysis_data.corr()["error"]
        .drop("error")
        .sort_values(key=abs, ascending=False)
    )
    print(correlations)

    # 8. Submission
    THRESHOLD = 0.05442899838089943

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Predict on test set
        raw_test_preds, test_ids = predict(model, test_loader, device)

        # Inverse transform (log1p -> expm1)
        final_test_preds = inverse_transform(raw_test_preds)

        # Save
        generate_submission(final_test_preds, test_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
