import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from library.config import Config
from library.data import get_dataloaders
from library.model import IS_RA_CGN
from library.utils import set_seed, TargetScaler


def run_training(config=None):
    """
    Main training function.

    Args:
        config (Config, optional): Configuration object. If None, uses default Config.
    """
    if config is None:
        config = Config()

    # Ensure reproducibility
    set_seed(config.seed)

    # 1. Prepare Data
    # load_cached_data=True will use existing .npz files if available
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=config.batch_size
    )

    # 2. Initialize Model and Components
    model = IS_RA_CGN(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    criterion = nn.MSELoss()

    # 3. Fit Target Scaler
    print("Fitting target scaler on training data...")
    scaler = TargetScaler()
    all_targets = []
    for data in train_loader:
        all_targets.append(data.y)

    if len(all_targets) > 0:
        all_targets = torch.cat(all_targets, dim=0)
        scaler.fit(all_targets)
        print(f"Targets Mean: {scaler.mean}")
        print(f"Targets Std: {scaler.std}")
    else:
        print("Warning: No training data found to fit scaler.")
        return

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {config.num_epochs} epochs...")

    for epoch in range(config.num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for data in train_loader:
            data = data.to(config.device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # Scale targets for loss computation
            targets_scaled = scaler.transform(data.y)

            loss = criterion(outputs, targets_scaled)
            loss.backward()
            optimizer.step()

            # Accumulate loss (MSE is mean, so multiply by batch size)
            train_loss += loss.item() * data.num_graphs
            total_train_samples += data.num_graphs

        avg_train_loss = (
            train_loss / total_train_samples if total_train_samples > 0 else 0.0
        )

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_mae_formation = 0.0
        val_mae_bandgap = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(config.device)
                outputs = model(data)

                # Loss on scaled targets
                targets_scaled = scaler.transform(data.y)
                loss = criterion(outputs, targets_scaled)
                val_loss += loss.item() * data.num_graphs

                # Metrics on original scale
                preds_orig = scaler.inverse_transform(outputs)
                targets_orig = data.y

                mae = torch.abs(preds_orig - targets_orig)
                # Sum MAE for each target separately
                val_mae_formation += mae[:, 0].sum().item()
                val_mae_bandgap += mae[:, 1].sum().item()

                total_val_samples += data.num_graphs

        avg_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0.0
        avg_val_mae_formation = (
            val_mae_formation / total_val_samples if total_val_samples > 0 else 0.0
        )
        avg_val_mae_bandgap = (
            val_mae_bandgap / total_val_samples if total_val_samples > 0 else 0.0
        )

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Logging
        print(
            f"Epoch {epoch+1:03d}: Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, "
            f"Val MAE Form: {avg_val_mae_formation}, Val MAE Band: {avg_val_mae_bandgap}"
        )

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model and scaler
            torch.save(
                model.state_dict(),
                os.path.join(config.checkpoint_dir, "best_model.pth"),
            )
            scaler.save(os.path.join(config.cache_dir, "target_scaler.npz"))
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Generate Submission
    print("Generating submission...")

    # Load best model and scaler
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    scaler_path = os.path.join(config.cache_dir, "target_scaler.npz")

    if os.path.exists(best_model_path) and os.path.exists(scaler_path):
        model.load_state_dict(torch.load(best_model_path))
        scaler.load(scaler_path)
    else:
        print("Warning: Best model or scaler not found. Using current model state.")

    model.eval()

    ids = []
    preds_formation = []
    preds_bandgap = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(config.device)
            outputs = model(data)

            # Inverse transform to get original scale (eV)
            preds_orig = scaler.inverse_transform(outputs)

            # Collect data
            # PyG batches custom attributes into lists
            if hasattr(data, "id"):
                ids.extend(data.id)
            else:
                # Fallback if id is not present (should not happen with library.data)
                print("Warning: 'id' attribute missing in test batch.")

            preds_formation.extend(preds_orig[:, 0].cpu().numpy())
            preds_bandgap.extend(preds_orig[:, 1].cpu().numpy())

    # Create submission dataframe
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds_formation,
            "bandgap_energy_ev": preds_bandgap,
        }
    )

    submission_path = os.path.join(config.submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
