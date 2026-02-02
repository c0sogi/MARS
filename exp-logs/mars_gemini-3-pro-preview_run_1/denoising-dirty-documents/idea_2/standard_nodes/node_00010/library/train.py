import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import get_dataloaders
from library.model import UNet
from library.utils import seed_everything, calculate_rmse, pad_image, unpad_image


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch on the training dataset.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        outputs = model(noisy)
        loss = criterion(outputs, clean)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def predict_tensor(model, image_tensor, tta=False):
    """
    Helper function to perform inference on a single image tensor.
    Supports Test-Time Augmentation (TTA) by averaging predictions
    of the original and geometric transformations (flips).

    Args:
        model: The trained PyTorch model.
        image_tensor: Input tensor of shape (1, 1, H, W).
        tta (bool): Whether to apply TTA.

    Returns:
        torch.Tensor: The predicted output tensor.
    """
    if not tta:
        return model(image_tensor)

    # TTA Strategy: Average predictions of Original, H-Flip, V-Flip, and HV-Flip
    # 1. Original
    pred_1 = model(image_tensor)

    # 2. Horizontal Flip
    img_h = torch.flip(image_tensor, dims=[3])
    out_h = model(img_h)
    pred_2 = torch.flip(out_h, dims=[3])

    # 3. Vertical Flip
    img_v = torch.flip(image_tensor, dims=[2])
    out_v = model(img_v)
    pred_3 = torch.flip(out_v, dims=[2])

    # 4. Rotate 180 (Horizontal + Vertical Flip)
    img_hv = torch.flip(image_tensor, dims=[2, 3])
    out_hv = model(img_hv)
    pred_4 = torch.flip(out_hv, dims=[2, 3])

    return (pred_1 + pred_2 + pred_3 + pred_4) / 4.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Calculates the average Root Mean Squared Error (RMSE).
    """
    model.eval()
    total_rmse = 0.0

    with torch.no_grad():
        for noisy, clean in loader:
            # Inputs are (1, 1, H, W) tensors
            # Convert to numpy to handle padding logic
            noisy_np = noisy.squeeze().numpy()
            clean_np = clean.squeeze().numpy()

            # Pad image to be divisible by 32 (required for U-Net pooling)
            padded_noisy = pad_image(noisy_np, factor=32)
            input_tensor = (
                torch.from_numpy(padded_noisy).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict (TTA disabled for validation speed)
            output_tensor = predict_tensor(model, input_tensor, tta=False)

            # Post-process: Unpad and Clip
            output_np = output_tensor.squeeze().cpu().numpy()
            output_clean = unpad_image(output_np, noisy_np.shape)
            output_clean = np.clip(output_clean, 0, 1)

            # Calculate RMSE
            rmse = calculate_rmse(clean_np, output_clean)
            total_rmse += rmse

    return total_rmse / len(loader)


def generate_submission(model, device):
    """
    Generates the submission CSV file for the test set.
    Loads the best model checkpoint and applies Test-Time Augmentation.
    """
    print("Generating submission...")

    # Load test loader
    _, _, test_loader = get_dataloaders()

    # Load best model weights
    checkpoint_path = Config.MODEL_CHECKPOINT
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    model.eval()
    submission_data = []

    with torch.no_grad():
        for noisy, img_ids in test_loader:
            img_id = img_ids[0]
            noisy_np = noisy.squeeze().numpy()

            # Pad image
            padded_noisy = pad_image(noisy_np, factor=32)
            input_tensor = (
                torch.from_numpy(padded_noisy).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict with TTA enabled
            output_tensor = predict_tensor(model, input_tensor, tta=Config.TTA_ENABLED)

            # Post-process
            output_np = output_tensor.squeeze().cpu().numpy()
            output_clean = unpad_image(output_np, noisy_np.shape)
            output_clean = np.clip(output_clean, 0, 1)

            # Format output for CSV: id_row_col, value
            h, w = output_clean.shape
            rows, cols = np.indices((h, w))
            # 1-based indexing
            rows += 1
            cols += 1

            flat_vals = output_clean.flatten()
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()

            # Generate IDs efficiently
            prefix = f"{img_id}_"
            ids = [f"{prefix}{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

            submission_data.extend(zip(ids, flat_vals))

    # Save to CSV
    df = pd.DataFrame(submission_data, columns=["id", "value"])
    df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    patience=50,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    t_max=Config.T_MAX,
    eta_min=Config.ETA_MIN,
):
    """
    Main execution function.
    Sets up the environment, trains the model, and generates the submission.

    Args:
        num_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience epochs.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        t_max (int): T_max for Cosine Annealing scheduler.
        eta_min (float): Minimum learning rate for scheduler.

    Returns:
        float: The best validation RMSE achieved.
    """
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize U-Net
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES,
    ).to(device)

    # Optimization Setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )

    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {num_epochs} epochs.")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse}"
        )

        # Checkpoint and Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best RMSE: {best_rmse}"
            )
            break

    print(f"Training finished. Best RMSE: {best_rmse}")

    # Generate final submission using the best model
    generate_submission(model, device)

    return best_rmse
