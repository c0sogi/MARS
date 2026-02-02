import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.utils import seed_everything, rle_encode, unpad_image, calc_map
from library.losses import SaltLoss
from library.model import ResNet34WideLinkNet
from library.dataset import get_dataloaders, get_test_loader

# Constants
WORKING_DIR = "./working/idea_33"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    Applies Gaussian noise to depth for regularization (Depth Jitter).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, masks, depths, ids) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        # Depth Jitter: Add Gaussian noise to normalized depth
        # sigma = 0.1 as per strategy
        noise = torch.randn_like(depths) * 0.1
        noisy_depths = depths + noise

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, noisy_depths)

        # Loss calculation
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Unpads predictions and targets to 101x101 before calculating mAP.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, depths, ids in dataloader:
            images = images.to(device)
            depths = depths.to(device)

            # Forward pass
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # Move to CPU for processing
            probs_np = probs.detach().cpu().numpy()  # (B, 1, 128, 128)
            masks_np = masks.detach().cpu().numpy()  # (B, 1, 128, 128)

            # Unpad each image in the batch back to 101x101
            # Squeeze channel dim: (B, 128, 128)
            probs_np = probs_np.squeeze(1)
            masks_np = masks_np.squeeze(1)

            batch_preds = []
            batch_targets = []

            for i in range(probs_np.shape[0]):
                p_unpadded = unpad_image(probs_np[i], original_size=101)
                m_unpadded = unpad_image(masks_np[i], original_size=101)
                batch_preds.append(p_unpadded)
                batch_targets.append(m_unpadded)

            all_preds.append(np.array(batch_preds))
            all_targets.append(np.array(batch_targets))

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate mAP
    score = calc_map(all_preds, all_targets)
    return score


def predict_marginalized(model, dataloader, device, num_scans=10):
    """
    Performs Marginalized Depth-Scan Inference on the test set.
    Scans a range of depth values and averages the predictions.
    """
    model.eval()
    results = []
    ids_list = []

    # Define scan range: -2.0 to 2.0 standard deviations
    scan_values = np.linspace(-2.0, 2.0, num_scans)

    with torch.no_grad():
        for images, _, _, ids in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            # Accumulator for marginalized probabilities
            # Shape: (B, 1, 128, 128)
            accum_probs = torch.zeros((batch_size, 1, 128, 128), device=device)

            # Marginalization Loop
            for z_val in scan_values:
                # Create constant depth tensor for this scan step
                z_tensor = torch.full(
                    (batch_size,), z_val, device=device, dtype=torch.float32
                )

                logits = model(images, z_tensor)
                probs = torch.sigmoid(logits)
                accum_probs += probs

            # Average over scan steps
            avg_probs = accum_probs / num_scans

            # Process batch
            avg_probs_np = avg_probs.detach().cpu().numpy().squeeze(1)

            for i in range(batch_size):
                # Unpad to 101x101
                pred_101 = unpad_image(avg_probs_np[i], original_size=101)

                # Binarize (Threshold 0.5)
                mask_bin = (pred_101 > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask_bin)

                results.append(rle)
                ids_list.append(ids[i])

    return ids_list, results


def train_and_evaluate(
    num_epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-2, patience=10
):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    seed_everything(42)

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    print("Initializing dataloaders...")
    train_loader, val_loader, depth_stats = get_dataloaders(batch_size=batch_size)

    # Model Initialization
    print("Initializing model...")
    model = ResNet34WideLinkNet(pretrained=True)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = SaltLoss(bce_weight=1.0, lovasz_weight=1.0)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training Loop
    best_map = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_map = evaluate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val mAP: {val_map:.10f}"
        )

        # Checkpointing and Early Stopping
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"New best mAP! Model saved to {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
            )
            break

    print(f"Training complete. Best Val mAP: {best_map:.10f}")
    return best_model_path, depth_stats


def generate_submission(model_path, depth_stats, batch_size=32):
    """
    Loads the best model and generates the submission file using Marginalized Inference.
    """
    print("Generating submission...")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = ResNet34WideLinkNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Get Test Loader
    test_loader = get_test_loader(batch_size=batch_size, depth_stats=depth_stats)

    # Predict using Marginalization
    ids, rles = predict_marginalized(model, test_loader, device, num_scans=10)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "rle_mask": rles})

    # Sort by ID (optional but good practice)
    df_sub = df_sub.sort_values("id")

    # Save
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
