import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.utils import set_seed, get_device
from library.model import CactusResNet
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle cases with single class in batch
        auc = 0.5

    return epoch_loss, auc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def train_single_seed(seed, train_loader, val_loader, device, epochs=20, patience=5):
    """
    Trains a single model instance with a specific seed.
    Manages Optimizer, Scheduler, and Early Stopping.
    """
    set_seed(seed)
    print(f"\n--- Training Seed {seed} ---")

    # Initialize Model
    model = CactusResNet(num_classes=1).to(device)

    # Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} AUC: {train_auc} | Val Loss: {val_loss} AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def generate_submission(
    models, test_loader, device, output_path="submission/submission.csv"
):
    """
    Generates predictions using the ensemble of models and Test Time Augmentation (TTA).
    Saves the result to a CSV file.
    """
    print("\nGenerating submission with TTA...")

    for model in models:
        model.eval()

    results = {}

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Accumulator for predictions
            batch_preds_sum = None

            # TTA Variants: Original, Horizontal Flip, Vertical Flip
            tta_variants = [
                images,
                torch.flip(images, [3]),  # Horizontal
                torch.flip(images, [2]),  # Vertical
            ]

            for img_variant in tta_variants:
                for model in models:
                    # Forward pass
                    out = torch.sigmoid(model(img_variant))
                    preds = out.cpu().numpy().flatten()

                    if batch_preds_sum is None:
                        batch_preds_sum = preds
                    else:
                        batch_preds_sum += preds

            # Calculate Arithmetic Mean
            # Total predictions = (Number of TTA variants) * (Number of Models)
            num_predictions = len(tta_variants) * len(models)
            avg_preds = batch_preds_sum / num_predictions

            # Store results
            for img_id, pred in zip(ids, avg_preds):
                results[img_id] = pred

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(list(results.items()), columns=["id", "has_cactus"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_experiment(epochs=20, batch_size=64, seeds=[0, 1, 2, 3, 4], debug_size=None):
    """
    Main driver function to run the full training and inference pipeline.
    """
    device = get_device()

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, debug_size=debug_size
    )

    # Train Ensemble
    trained_models = []
    for seed in seeds:
        model = train_single_seed(seed, train_loader, val_loader, device, epochs=epochs)
        trained_models.append(model)

    # Generate Submission
    generate_submission(trained_models, test_loader, device)
