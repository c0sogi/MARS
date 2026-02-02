import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from library.config import Config
from library.utils import mixup_data, mixup_criterion, calculate_multilabel_auc


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Performs one epoch of training with Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        images, labels_a, labels_b, lam = mixup_data(
            images, labels, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        auc = calculate_multilabel_auc(all_labels, all_preds)
    else:
        auc = 0.5

    return auc, avg_loss


def run_inference(model, loader, device, tta=False):
    """
    Generates predictions for the test set, optionally using TTA (Horizontal Flip).
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for images, rec_ids in loader:
            images = images.to(device)

            # Forward pass 1: Original
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            if tta:
                # Forward pass 2: Horizontal Flip
                # Images are (B, C, H, W). Flip on W dimension (dim 3)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(outputs_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            probs_np = probs.cpu().numpy()
            rec_ids_np = rec_ids.numpy()

            for i in range(len(rec_ids_np)):
                results[rec_ids_np[i]] = probs_np[i]

    return results


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    swa_start_epoch,
    patience=10,
):
    """
    Main training loop with SWA and Early Stopping.
    """
    criterion = nn.BCEWithLogitsLoss()

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    best_auc = 0.0
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # To handle the case where SWA is never triggered (e.g. short run)
    final_model_is_swa = False

    print(
        f"Starting training for {epochs} epochs. SWA starts at epoch {swa_start_epoch}."
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_auc, val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # SWA Logic
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            final_model_is_swa = True
            # Disable early stopping during SWA
        else:
            # Standard Early Stopping / Model Checkpointing logic
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Finalize Model
    if final_model_is_swa:
        print("Updating SWA BatchNorm statistics...")
        # update_bn expects the loader to yield batch[0] as input.
        # Our loader yields (img, label, id), so batch[0] is img. This works.
        update_bn(train_loader, swa_model, device=device)
        return swa_model
    else:
        print("Restoring best model from early stopping phase...")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model


def save_submission(predictions, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions (dict): Dictionary mapping rec_id to probability array (19 classes).
        output_path (str): Path to save the CSV.
    """
    data = []

    # Sort by rec_id to ensure consistent order (though not strictly required)
    sorted_ids = sorted(predictions.keys())

    for rec_id in sorted_ids:
        probs = predictions[rec_id]
        for species_idx, prob in enumerate(probs):
            # Format: Id = rec_id * 100 + species_idx
            row_id = int(rec_id * 100 + species_idx)
            data.append({"Id": row_id, "Probability": prob})

    df = pd.DataFrame(data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
