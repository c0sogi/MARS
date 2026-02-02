import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, rmsle
from library.data import get_dataloaders
from library.model import ACC_WDS
from library.train import train_one_epoch, validate


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Override Config for fast baseline execution
    # Reducing epochs to ensure completion within time limit while allowing convergence
    Config.NUM_EPOCHS = 50

    # 2. Prepare Data
    # load_cached_data=True will use the preprocessed .npz files if they exist
    print("Initializing data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    model = ACC_WDS().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss Function (MSE on log-transformed targets)
    criterion = nn.MSELoss()

    # 4. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmsle = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training complete.")

    # 5. Final Validation and Failure Analysis
    print("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_global_feats_list = []

    with torch.no_grad():
        for batch in val_loader:
            af = batch["atomic_features"].to(device)
            gf = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            outputs = model(af, gf, mask)

            # Inverse transform: log(1+x) -> exp(x) - 1
            preds_original = torch.expm1(outputs)
            targets_original = torch.expm1(target)

            val_preds_list.append(preds_original.cpu().numpy())
            val_targets_list.append(targets_original.cpu().numpy())
            val_global_feats_list.append(gf.cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)
    val_global_feats = np.concatenate(val_global_feats_list, axis=0)

    # Compute Final Metric
    final_metric = rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate Mean Absolute Error per sample (averaged over the two targets)
    # This gives a magnitude of error for correlation analysis
    sample_errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Global feature names corresponding to Config.GLOBAL_FEATURE_DIM
    # 0-2: Lattice lengths, 3-5: Angles, 6: Volume, 7: Density, 8-10: Stoich (Al, Ga, In), 11: Total Atoms
    feature_names = [
        "lat_a",
        "lat_b",
        "lat_c",
        "alpha",
        "beta",
        "gamma",
        "volume",
        "density",
        "Al_frac",
        "Ga_frac",
        "In_frac",
        "total_atoms",
    ]

    df_analysis = pd.DataFrame(val_global_feats, columns=feature_names)
    df_analysis["error_magnitude"] = sample_errors

    print(
        "\nFailure Analysis (Correlation between Global Features and Error Magnitude):"
    )
    correlations = (
        df_analysis.corr()["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )
    print(correlations)

    # 6. Submission Generation
    threshold = 0.05479004207787702

    if final_metric < threshold:
        print("\nMetric passed threshold. Generating submission...")
        test_preds_list = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                af = batch["atomic_features"].to(device)
                gf = batch["global_features"].to(device)
                mask = batch["mask"].to(device)
                ids = batch["id"]

                outputs = model(af, gf, mask)
                preds_original = torch.expm1(outputs)

                test_preds_list.append(preds_original.cpu().numpy())
                test_ids_list.extend(ids.numpy())

        test_preds = np.concatenate(test_preds_list, axis=0)

        submission_df = pd.DataFrame(
            {
                "id": test_ids_list,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_metric} is not lower than threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
