import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.utils import calculate_metric
from library.data_loader import MelanomaDataset, get_transforms
from library.config import SUBMISSION_PATH


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, tabular, targets in loader:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images, tabular)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, tabular)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = running_loss / dataset_size
    auc = calculate_metric(np.array(all_targets), np.array(all_preds))

    return avg_loss, auc


def predict_tta(model, test_df, test_tab, tta_steps, batch_size, num_workers, device):
    """
    Performs Test-Time Augmentation (TTA) prediction.
    1. Predicts on standard test set.
    2. Predicts on augmented test set `tta_steps` times.
    3. Averages the results.
    """
    model.eval()

    def get_preds(loader):
        preds = []
        with torch.no_grad():
            for images, tabular, _ in loader:
                images = images.to(device)
                tabular = tabular.to(device)

                logits = model(images, tabular)
                probs = torch.sigmoid(logits)
                preds.extend(probs.cpu().numpy().flatten())
        return np.array(preds)

    # 1. Standard Prediction
    # Use 'test' transforms (Resize + Normalize only)
    test_dataset = MelanomaDataset(
        test_df, test_tab, transform=get_transforms("test"), is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Predicting on standard test set...")
    final_preds = get_preds(test_loader)

    # 2. TTA Prediction
    if tta_steps > 0:
        print(f"Performing TTA with {tta_steps} steps...")

        # Use 'train' transforms (Augmentations enabled)
        tta_dataset = MelanomaDataset(
            test_df, test_tab, transform=get_transforms("train"), is_test=True
        )
        tta_loader = DataLoader(
            tta_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        tta_accum = np.zeros_like(final_preds)

        for i in range(tta_steps):
            preds = get_preds(tta_loader)
            tta_accum += preds

        # Average: (Standard + Sum(TTA)) / (1 + TTA_STEPS)
        final_preds = (final_preds + tta_accum) / (1 + tta_steps)

    return final_preds


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    save_path,
    scheduler=None,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )
        print(f"Validation AUC: {val_auc}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"AUC improved. Saved model to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(image_names, predictions, output_path=SUBMISSION_PATH):
    """
    Saves predictions to a CSV file in the required format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({"image_name": image_names, "target": predictions})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
