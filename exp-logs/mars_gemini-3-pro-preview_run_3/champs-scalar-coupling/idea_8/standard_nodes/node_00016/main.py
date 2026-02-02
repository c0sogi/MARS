import os
import torch
import numpy as np
import pandas as pd
import random
import sys
from library.config import Config
from library.trainer import ModelTrainer
from library.data import get_data_loaders

# ==========================================
# Configuration Override for Fast Baseline
# ==========================================
# Limit execution time and resource usage
Config.DEBUG = True
Config.DEBUG_SAMPLES = 25000  # Process a subset of data
Config.MAX_EPOCHS = 5  # Limit epochs
Config.BATCH_SIZE = 64  # Ensure GPU memory safety


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    print("Initializing Fast Baseline Run...")

    # Initialize Trainer
    # We use debug=True to trigger the sample limiting in the Dataset class
    trainer = ModelTrainer(debug=Config.DEBUG, load_cached_data=True)

    # Get Loaders explicitly to manage the loop
    train_loader, val_loader, test_loader, scaler = get_data_loaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=True
    )

    # Assign scaler to trainer (usually done inside trainer.run, but we are running manually)
    trainer.scaler = scaler

    # 2. Training Loop
    print(
        f"Starting training for {Config.MAX_EPOCHS} epochs on device {Config.DEVICE}..."
    )
    best_score = float("inf")

    for epoch in range(Config.MAX_EPOCHS):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_score = trainer.validate(val_loader)

        # Scheduler Step
        trainer.scheduler.step()
        current_lr = trainer.scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Score: {val_score:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved! Score: {best_score:.6f}")

    # 3. Final Validation & Metric Calculation
    print("\nPerforming Final Validation Assessment...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )

    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_types = []
    all_dists = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(Config.DEVICE)

            # Forward
            pred_c = trainer.model(batch)

            # Inverse Transform
            pred_real = scaler.inverse_transform(pred_c, batch.coupling_type)
            target_real = batch.y

            # Calculate Pair Distances for Analysis
            # batch.pos: (N, 3), coupling_index: (2, K)
            pos = batch.pos
            idx0 = batch.coupling_index[0]
            idx1 = batch.coupling_index[1]
            dists = (pos[idx0] - pos[idx1]).norm(dim=1)

            # Store
            all_preds.append(pred_real.cpu().numpy())
            all_targets.append(target_real.cpu().numpy())
            all_types.append(batch.coupling_type.cpu().numpy())
            all_dists.append(dists.cpu().numpy())

    # Concatenate
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    types = np.concatenate(all_types)
    dists = np.concatenate(all_dists)

    # Calculate Metric: Mean(Log(MAE_type))
    unique_types = np.unique(types)
    log_maes = []

    for t_idx in unique_types:
        mask = types == t_idx
        mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
        log_mae = np.log(mae + 1e-9)
        log_maes.append(log_mae)

    final_metric = np.mean(log_maes)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    abs_errors = np.abs(y_pred - y_true)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {"error": abs_errors, "distance": dists, "target_magnitude": np.abs(y_true)}
    )

    # Correlations
    corr_dist = df_analysis["error"].corr(df_analysis["distance"])
    corr_mag = df_analysis["error"].corr(df_analysis["target_magnitude"])

    print(f"Correlation (Error vs Distance): {corr_dist:.4f}")
    print(f"Correlation (Error vs Target Magnitude): {corr_mag:.4f}")

    # 5. Conditional Submission
    THRESHOLD = -1.2761284112930298

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.4f}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_and_submit(test_loader)
    else:
        print(
            f"\nMetric ({final_metric:.4f}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
