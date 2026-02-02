import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library import config, utils, data, model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, ages, implants, labels) in enumerate(dataloader):
        images = images.to(device)
        ages = ages.to(device)
        implants = implants.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape is (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, ages, implants)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping is explicitly disabled as per configuration
        if config.USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and pF1 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, ages, implants, labels in dataloader:
            images = images.to(device)
            ages = ages.to(device)
            implants = implants.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            logits = model(images, ages, implants)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate pF1 score
    pf1 = utils.pf1_score(all_labels, all_preds)

    return epoch_loss, pf1


def predict_and_submit(model_path, device):
    """
    Loads the best model, generates predictions on the test set,
    aggregates them by prediction_id (taking the max), and saves to CSV.
    """
    print("Starting inference and submission generation...")

    # Load Model
    net = model.MetadataEfficientNet()
    net.to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using initialized weights.")

    net.eval()

    # Get Test Loader (and metadata for mapping IDs)
    # We pass load_cached_data=True to reuse the parquet files generated during training
    _, _, test_loader = data.get_dataloaders(load_cached_data=True)

    # We need the test dataframe to map predictions back to prediction_id
    # Since the loader is sequential (shuffle=False), the order matches the dataframe
    df_test = pd.read_parquet(
        os.path.join(config.WORKING_DIR, "processed_test.parquet")
    )

    all_probs = []

    with torch.no_grad():
        for images, ages, implants in test_loader:
            images = images.to(device)
            ages = ages.to(device)
            implants = implants.to(device)

            # Forward pass
            # Note: Modality Dropout is disabled in eval mode automatically
            logits = net(images, ages, implants)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    # Flatten predictions
    flat_probs = np.concatenate(all_probs).flatten()

    # Assign predictions to dataframe
    # Ensure lengths match
    if len(flat_probs) != len(df_test):
        print(
            f"Warning: Number of predictions ({len(flat_probs)}) does not match metadata length ({len(df_test)})."
        )
        # Truncate or pad if necessary (though this shouldn't happen with correct loaders)
        min_len = min(len(flat_probs), len(df_test))
        df_test = df_test.iloc[:min_len]
        flat_probs = flat_probs[:min_len]

    df_test["cancer_prob"] = flat_probs

    # Aggregate by prediction_id
    # Strategy: Max probability across images for the same breast (prediction_id)
    submission_df = df_test.groupby("prediction_id")["cancer_prob"].max().reset_index()

    # Rename columns to match submission format
    submission_df.rename(columns={"cancer_prob": "cancer"}, inplace=True)

    # Save submission
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(submission_df.head())


def run_training(epochs=config.EPOCHS, load_cached_data=False):
    """
    Main execution function for training the model.
    """
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = data.get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Model
    net = model.MetadataEfficientNet()
    net.to(device)

    # Loss Function (Weighted BCE)
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_pf1 = -1.0

    print("Starting training...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_pf1 = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val pF1: {val_pf1}"
        )

        # Save Best Model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with pF1: {best_pf1}")

    print(f"Training complete. Best Validation pF1: {best_pf1}")

    # Generate Submission
    predict_and_submit(config.MODEL_SAVE_PATH, device)
