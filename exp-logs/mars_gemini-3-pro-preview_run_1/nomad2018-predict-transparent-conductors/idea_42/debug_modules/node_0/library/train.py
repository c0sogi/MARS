import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    FACTOR,
    MIN_LR,
    ATOM_FEATURES_DIM,
    GLOBAL_FEATURES_DIM,
    HIDDEN_DIM,
    ATOMIC_LAYERS,
    GLOBAL_LAYERS,
    FUSION_LAYERS,
    DROPOUT,
    USE_BATCH_NORM,
    SEED,
)
from library.data import process_data, get_scalers, CrystalDataset, collate_sparse
from library.model import REMSWDSModel

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        atomic_features = batch["atomic_features"].to(device)
        batch_index = batch["batch_index"].to(device)
        global_features = batch["global_features"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(atomic_features, batch_index, global_features)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            atomic_features = batch["atomic_features"].to(device)
            batch_index = batch["batch_index"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, batch_index, global_features)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

    return running_loss / len(loader.dataset)


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            atomic_features = batch["atomic_features"].to(device)
            batch_index = batch["batch_index"].to(device)
            global_features = batch["global_features"].to(device)
            ids = batch["ids"]

            outputs = model(atomic_features, batch_index, global_features)

            # Inverse transform: exp(x) - 1 to revert log1p
            preds = torch.expm1(outputs)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(ids.numpy())

    return np.vstack(all_preds), np.concatenate(all_ids)


def train_model():
    """
    Main function to manage the training lifecycle.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Process Data (Features Extraction)
    # This step uses caching to avoid re-computing features if they exist
    print("Processing data...")
    train_af, train_gf, train_y, train_ids = process_data(train_df, TRAIN_CACHE_PATH)
    val_af, val_gf, val_y, val_ids = process_data(val_df, VAL_CACHE_PATH)
    test_af, test_gf, test_y, test_ids = process_data(test_df, TEST_CACHE_PATH)

    # 3. Fit Scalers
    print("Fitting scalers on training data...")
    scaler_atomic, scaler_global = get_scalers(train_af, train_gf)

    # 4. Create Datasets
    print("Creating datasets...")
    train_dataset = CrystalDataset(
        train_af,
        train_gf,
        train_y,
        train_ids,
        scaler_atomic=scaler_atomic,
        scaler_global=scaler_global,
        mode="train",
    )
    val_dataset = CrystalDataset(
        val_af,
        val_gf,
        val_y,
        val_ids,
        scaler_atomic=scaler_atomic,
        scaler_global=scaler_global,
        mode="val",
    )
    test_dataset = CrystalDataset(
        test_af,
        test_gf,
        test_y,
        test_ids,
        scaler_atomic=scaler_atomic,
        scaler_global=scaler_global,
        mode="test",
    )

    # 5. Create DataLoaders
    # Use collate_sparse to handle variable number of atoms per crystal
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sparse,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse,
        num_workers=0,
    )

    # 6. Initialize Model
    print("Initializing model...")
    model = REMSWDSModel(
        atom_features_dim=ATOM_FEATURES_DIM,
        global_features_dim=GLOBAL_FEATURES_DIM,
        hidden_dim=HIDDEN_DIM,
        atomic_layers=ATOMIC_LAYERS,
        global_layers=GLOBAL_LAYERS,
        fusion_layers=FUSION_LAYERS,
        dropout=DROPOUT,
        use_bn=USE_BATCH_NORM,
    ).to(device)

    # 7. Setup Training Components
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=FACTOR, patience=10, min_lr=MIN_LR, verbose=False
    )

    # 8. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # RMSLE is sqrt of MSE on log-transformed data
        train_rmsle = np.sqrt(train_loss)
        val_rmsle = np.sqrt(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.8f} (RMSLE: {train_rmsle:.8f}) | "
            f"Val Loss: {val_loss:.8f} (RMSLE: {val_rmsle:.8f})"
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved! (Val Loss: {val_loss:.8f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # 9. Inference and Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    print("Generating predictions on test set...")
    predictions, ids = predict(model, test_loader, device)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Save submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    return best_val_loss
