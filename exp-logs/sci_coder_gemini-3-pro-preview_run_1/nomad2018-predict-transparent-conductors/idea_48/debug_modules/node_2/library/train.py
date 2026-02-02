import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time

from library.config import (
    WORKING_DIR,
    SUBMISSION_DIR,
    EARLY_STOPPING_PATIENCE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
)

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    device=None,
    patience=EARLY_STOPPING_PATIENCE,
):
    """
    Training loop with Early Stopping and Scheduler.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {device}")
    model = model.to(device)

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Loss Function: MSE (targets are already log1p transformed)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pt")

    start_time = time.time()

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch_idx, (
            batch_atomic,
            batch_index,
            batch_global,
            batch_targets,
            _,
        ) in enumerate(train_loader):
            batch_atomic = batch_atomic.to(device)
            batch_index = batch_index.to(device)
            batch_global = batch_global.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            outputs = model(batch_atomic, batch_index, batch_global)
            loss = criterion(outputs, batch_targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_targets.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_atomic, batch_index, batch_global, batch_targets, _ in val_loader:
                batch_atomic = batch_atomic.to(device)
                batch_index = batch_index.to(device)
                batch_global = batch_global.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_atomic, batch_index, batch_global)
                loss = criterion(outputs, batch_targets)

                val_loss += loss.item() * batch_targets.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        # --- Logging ---
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.8f} | "
            f"Val Loss: {avg_val_loss:.8f}"
        )

        # --- Scheduler Step ---
        scheduler.step(avg_val_loss)

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.8f}")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model weights.")

    return model


def generate_submission(model, test_loader, output_path=None, device=None):
    """
    Generates predictions for the test set and saves to CSV.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_path is None:
        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    model.eval()
    model.to(device)

    all_ids = []
    all_preds = []

    print("Generating predictions...")

    with torch.no_grad():
        for batch_atomic, batch_index, batch_global, _, batch_ids in test_loader:
            batch_atomic = batch_atomic.to(device)
            batch_index = batch_index.to(device)
            batch_global = batch_global.to(device)

            # Forward pass
            outputs = model(batch_atomic, batch_index, batch_global)

            # Inverse transform: log1p -> expm1
            # The targets were transformed using np.log1p, so we use torch.expm1 to reverse it.
            # Ensure non-negative predictions if physically required, though expm1 handles this naturally for log space.
            preds = torch.expm1(outputs)

            all_ids.extend(batch_ids.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Create DataFrame
    # Columns: id, formation_energy_ev_natom, bandgap_energy_ev
    df = pd.DataFrame(
        all_preds, columns=["formation_energy_ev_natom", "bandgap_energy_ev"]
    )
    df.insert(0, "id", all_ids)

    # Sort by ID to be safe
    df = df.sort_values("id")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return df
