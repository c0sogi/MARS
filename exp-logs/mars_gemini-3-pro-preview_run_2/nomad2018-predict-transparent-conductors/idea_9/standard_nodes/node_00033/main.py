import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_log_error

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import DualStreamCGCNN
from library.train import train_epoch, validate, set_seed


def generate_submission(model, test_loader, target_scaler):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)
    model.eval()

    ids = []
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # Forward pass
            out = model(batch)
            # Inverse transform to original scale
            pred_orig = target_scaler.inverse_transform(out)

            ids.extend(batch.id.cpu().numpy().flatten())
            preds.append(pred_orig.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Ensure non-negative predictions for RMSLE compatibility (physics constraint)
    preds = np.maximum(preds, 0)

    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Sort by ID
    df = df.sort_values("id")

    # Save
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    # 1. Configure for Fast Baseline
    # Adjust hyperparameters to ensure execution finishes within time limits
    Config.MAX_EPOCHS = 80
    Config.PATIENCE = 15
    Config.BATCH_SIZE = 64

    # 2. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 3. Data Loading
    print("Loading and processing data...")
    # load_cached_data=True will try to use existing .npz files in ./working/idea_9/cache
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=True
    )

    # 4. Model Initialization
    print("Initializing Dual-Stream CGCNN model...")
    model = DualStreamCGCNN().to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 5. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # 6. Final Validation & Metric Calculation
    print("Loading best model for final validation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model.eval()

    val_preds = []
    val_targets = []
    val_global_feats = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)

            # Inverse transform to get real units (eV)
            pred_original = target_scaler.inverse_transform(out)
            true_original = target_scaler.inverse_transform(batch.y)

            val_preds.append(pred_original.cpu().numpy())
            val_targets.append(true_original.cpu().numpy())
            # Collect global features for failure analysis
            val_global_feats.append(batch.global_feat.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_global_feats = np.concatenate(val_global_feats, axis=0)

    # Clip predictions to 0 to avoid error in log (physically energy >= 0 approx)
    val_preds_clipped = np.maximum(val_preds, 0)
    val_targets_clipped = np.maximum(val_targets, 0)

    # Calculate Column-wise Root Mean Squared Logarithmic Error
    rmsle_formation = np.sqrt(
        mean_squared_log_error(val_targets_clipped[:, 0], val_preds_clipped[:, 0])
    )
    rmsle_bandgap = np.sqrt(
        mean_squared_log_error(val_targets_clipped[:, 1], val_preds_clipped[:, 1])
    )
    final_metric = (rmsle_formation + rmsle_bandgap) / 2.0

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Performing failure analysis...")
    # Mean Absolute Error per sample
    errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Feature names corresponding to global_feat construction in library.data
    # [a, b, c, alpha, beta, gamma, frac_O, frac_Al, frac_Ga, frac_In]
    feature_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Frac_O",
        "Frac_Al",
        "Frac_Ga",
        "Frac_In",
    ]

    print("Correlation between Error Magnitude and Global Features:")
    correlations = {}
    for i, name in enumerate(feature_names):
        if i < val_global_feats.shape[1]:
            # Compute correlation
            if np.std(val_global_feats[:, i]) > 1e-6:  # Avoid constant features
                corr = np.corrcoef(errors, val_global_feats[:, i])[0, 1]
                correlations[name] = corr
            else:
                correlations[name] = 0.0

    # Sort and print
    for name, corr in sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {name}: {corr:.4f}")

    # 8. Submission Generation
    threshold = 0.05085437756413089
    if final_metric < threshold:
        print(
            f"Metric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        generate_submission(model, test_loader, target_scaler)
    else:
        print(
            f"Metric {final_metric} is NOT below threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
