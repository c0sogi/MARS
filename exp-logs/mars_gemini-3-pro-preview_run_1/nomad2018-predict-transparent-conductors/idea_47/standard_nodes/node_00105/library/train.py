import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.model import CEADSModel
from library.data import get_train_val_loaders, get_test_loader

# Set fixed seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch from SparseCollate
        atomic_feats, batch_indices, global_feats, targets, _ = batch

        # Move to device
        atomic_feats = atomic_feats.to(device)
        batch_indices = batch_indices.to(device)
        global_feats = global_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(atomic_feats, batch_indices, global_feats)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Multiply by batch size (number of crystals) to accumulate total loss
        running_loss += loss.item() * global_feats.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            atomic_feats, batch_indices, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            outputs = model(atomic_feats, batch_indices, global_feats)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * global_feats.size(0)

    return running_loss / len(loader.dataset)


def generate_submission(model, device, output_path="./submission/submission.csv"):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    # Using default batch size and workers for inference
    test_loader = get_test_loader(batch_size=64, num_workers=2)

    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_feats, batch_indices, global_feats, _, ids = batch

            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats = global_feats.to(device)

            # Predict (log space)
            outputs = model(atomic_feats, batch_indices, global_feats)

            # Inverse transform: exp(x) - 1
            preds = torch.expm1(outputs).cpu().numpy()
            ids = ids.numpy()

            for i in range(len(ids)):
                results.append(
                    {
                        "id": ids[i],
                        "formation_energy_ev_natom": preds[i, 0],
                        "bandgap_energy_ev": preds[i, 1],
                    }
                )

    # Create DataFrame and save
    df = pd.DataFrame(results)
    # Ensure correct column order
    df = df[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train(
    debug_size=None,
    epochs=150,
    batch_size=64,
    learning_rate=1e-3,
    patience=15,
    output_path="./submission/submission.csv",
):
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size, num_workers=2, debug_size=debug_size
    )

    # Model Initialization
    # Using default dimensions from library.model
    model = CEADSModel(
        atomic_input_dim=21,
        global_input_dim=22,
        atomic_hidden=512,
        global_hidden=256,
        fusion_hidden=256,
        output_dim=2,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.10f} - Val Loss: {val_loss:.10f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for submission
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val Loss: {best_val_loss:.10f}")

    # Generate Submission
    generate_submission(model, device, output_path=output_path)
