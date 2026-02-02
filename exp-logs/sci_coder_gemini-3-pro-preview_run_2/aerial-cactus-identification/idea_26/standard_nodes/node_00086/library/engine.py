import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.utils import set_seed, get_device, WORKING_DIR, SUBMISSION_DIR
from library.model import WideCoordinateResNeXt
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (N, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    avg_loss = running_loss / dataset_size

    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        # Handle case with single class in validation set (unlikely but safe)
        auc = 0.5

    return avg_loss, auc


def predict_with_tta(model, image_tensor, device):
    """
    Predicts with Test Time Augmentation: Original, Horizontal Flip, Vertical Flip.

    Args:
        model: The trained PyTorch model.
        image_tensor: Input image tensor of shape (C, H, W).
        device: Device to run on.

    Returns:
        float: Averaged probability.
    """
    model.eval()

    # image_tensor shape: (C, H, W)
    # Horizontal Flip (flip width, dim 2)
    img_h = torch.flip(image_tensor, [2])
    # Vertical Flip (flip height, dim 1)
    img_v = torch.flip(image_tensor, [1])

    # Stack into a batch: (3, C, H, W)
    batch = torch.stack([image_tensor, img_h, img_v]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)

    # Average the probabilities across the augmented versions
    return probs.mean().item()


def run_training_and_inference(
    epochs=20, batch_size=64, seeds=[0, 1, 2, 3, 4], patience=5
):
    """
    Main driver function to train the ensemble and generate submission.

    Args:
        epochs (int): Maximum number of training epochs per seed.
        batch_size (int): Batch size for dataloaders.
        seeds (list): List of random seeds for the ensemble.
        patience (int): Early stopping patience.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Get Test IDs for submission
    # test_loader.dataset is the CactusDataset, targets contains IDs for test set
    test_ids = test_loader.dataset.targets

    # Array to store accumulated predictions from all seeds
    final_preds = np.zeros(len(test_ids))

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)

        # Initialize Model
        model = WideCoordinateResNeXt(cardinality=32).to(device)

        # Loss and Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training Loop
        best_auc = 0.0
        best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model for inference
        print(f"Loading best model from {best_model_path} (AUC: {best_auc:.10f})")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Inference on Test Set with TTA
        print(f"Generating predictions for Seed {seed}...")
        seed_preds = []

        # Iterate over test set
        # test_loader batch_size is 1, so we process one image at a time
        for i, (images, _) in enumerate(test_loader):
            img = images.squeeze(0)  # (1, C, H, W) -> (C, H, W)
            prob = predict_with_tta(model, img, device)
            seed_preds.append(prob)

        final_preds += np.array(seed_preds)

    # Average predictions across seeds
    final_preds /= len(seeds)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(df_sub.head())
