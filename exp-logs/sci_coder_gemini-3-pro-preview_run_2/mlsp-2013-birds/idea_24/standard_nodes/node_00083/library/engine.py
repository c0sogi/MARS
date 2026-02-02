import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import calculate_roc_auc
from library.dataset import BirdDataset


def train_fn(model, data_loader, optimizer, device, scheduler, criterion):
    """
    Trains the model for one epoch using SAM optimizer.

    Args:
        model: PyTorch model.
        data_loader: DataLoader yielding (images, labels, rec_ids).
        optimizer: SAM optimizer instance.
        device: 'cuda' or 'cpu'.
        scheduler: Learning rate scheduler (optional).
        criterion: Loss function.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    final_loss = 0
    counter = 0

    for batch_idx, (images, labels, _) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # --- SAM Step 1 ---
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute gradients
        loss.backward()

        # First step: Perturb weights to maximize loss in neighborhood
        # zero_grad=True clears the gradients after using them to compute perturbation
        optimizer.first_step(zero_grad=True)

        # --- SAM Step 2 ---
        # Forward pass with perturbed weights
        outputs_2 = model(images)
        loss_2 = criterion(outputs_2, labels)

        # Compute gradients at perturbed state
        loss_2.backward()

        # Second step: Restore weights and update using gradients from perturbed state
        optimizer.second_step(zero_grad=True)

        # Step scheduler if it's step-based (though config uses Constant)
        if scheduler is not None:
            # Assuming scheduler might be step-based; if epoch-based, this is harmless
            # or handled outside. Given "Constant", this is likely a no-op or handled externally.
            pass

        final_loss += loss.item()
        counter += 1

    return final_loss / counter


def eval_fn(model, data_loader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        data_loader: DataLoader.
        device: 'cuda' or 'cpu'.
        criterion: Loss function.

    Returns:
        tuple: (average_loss, roc_auc_score, predictions, valid_labels)
    """
    model.eval()
    final_loss = 0
    counter = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            final_loss += loss.item()
            counter += 1

            # Apply sigmoid to logits to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if counter > 0:
        avg_loss = final_loss / counter
    else:
        avg_loss = 0.0

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    auc_score = calculate_roc_auc(all_labels, all_preds)

    return avg_loss, auc_score, all_preds, all_labels


def inference_fn(model, images, rec_ids, device):
    """
    Performs inference using Cyclic Test-Time Augmentation (TTA).

    Iterates through the shifts defined in Config.TTA_SHIFTS, creates a temporary
    dataset/loader for each shift, predicts, and averages the results.

    Args:
        model: PyTorch model.
        images: Numpy array of test images.
        rec_ids: Numpy array of recording IDs.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (rec_ids, averaged_probabilities)
    """
    model.eval()

    # Placeholder for accumulated probabilities
    # Shape: (N_samples, N_classes)
    accumulated_probs = np.zeros((len(images), Config.NUM_CLASSES), dtype=np.float32)

    # Dummy labels for creating the dataset (not used for inference)
    dummy_labels = np.zeros((len(images), Config.NUM_CLASSES), dtype=np.float32)

    print(f"Starting Inference with TTA Shifts: {Config.TTA_SHIFTS}")

    with torch.no_grad():
        for shift in Config.TTA_SHIFTS:
            # Create a dataset variant with the specific cyclic shift
            # split='test' ensures deterministic transforms (normalization only)
            # tta_shift triggers the np.roll logic in BirdDataset
            dataset = BirdDataset(
                images=images,
                labels=dummy_labels,
                rec_ids=rec_ids,
                split="test",
                tta_shift=shift,
            )

            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            fold_preds = []

            for batch_idx, (imgs, _, _) in enumerate(loader):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

            fold_preds = np.concatenate(fold_preds)

            # Add to accumulator
            accumulated_probs += fold_preds

    # Average across all TTA variants
    avg_probs = accumulated_probs / len(Config.TTA_SHIFTS)

    return rec_ids, avg_probs
