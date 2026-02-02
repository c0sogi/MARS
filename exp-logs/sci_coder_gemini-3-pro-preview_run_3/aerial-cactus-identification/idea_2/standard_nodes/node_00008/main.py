import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# Import provided library components
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CactusResNet
from library.train import train_one_epoch, validate, inference_tta
from library.utils import save_checkpoint, save_submission, load_checkpoint


def analyze_failures(model, loader, device):
    """
    Performs failure analysis by correlating prediction errors with
    image meta-features (brightness and contrast).
    """
    model.eval()
    errors = []
    brightness_vals = []
    contrast_vals = []

    # Parameters to reverse normalization
    # Normalization was (x - 0.5) / 0.5
    # Original = x * 0.5 + 0.5
    mean_tensor = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    print("Performing failure analysis on validation set...")

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)

            # Calculate absolute error
            batch_errors = torch.abs(probs - targets).cpu().numpy()
            errors.extend(batch_errors)

            # Revert normalization to get original image stats
            # We want stats of the original image content
            orig_images = images * std_tensor + mean_tensor

            # Compute Brightness (Mean intensity) and Contrast (Std intensity) per image
            # Shape: (B, 3, H, W) -> reduce over (1, 2, 3)
            b_batch = orig_images.mean(dim=(1, 2, 3)).cpu().numpy()
            c_batch = orig_images.std(dim=(1, 2, 3)).cpu().numpy()

            brightness_vals.extend(b_batch)
            contrast_vals.extend(c_batch)

    errors = np.array(errors)
    brightness_vals = np.array(brightness_vals)
    contrast_vals = np.array(contrast_vals)

    # Calculate correlations
    # numpy.corrcoef returns a matrix [[1, r], [r, 1]]
    corr_brightness = np.corrcoef(errors, brightness_vals)[0, 1]
    corr_contrast = np.corrcoef(errors, contrast_vals)[0, 1]

    print(f"Correlation between Error and Brightness: {corr_brightness:.6f}")
    print(f"Correlation between Error and Contrast: {corr_contrast:.6f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using debug=False to use full dataset as required for high performance
    dataloaders = get_dataloaders(debug=False)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 3. Model Initialization
    model = CactusResNet(num_classes=Config.NUM_CLASSES).to(device)

    # 4. Training Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Checkpoint path
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    best_auc = 0.0
    patience_counter = 0

    # 5. Training Loop
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("Training finished.")

    # 6. Evaluation
    # Load the best model weights
    model = load_checkpoint(model, best_model_path, device)

    # Compute final validation metric
    _, final_auc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Conditional Submission
    threshold = 0.9995999559487148
    if final_auc > threshold:
        print(
            f"Validation AUC ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Generate predictions using TTA
        predictions = inference_tta(model, test_loader, device)

        # Get IDs
        test_ids = test_loader.dataset.df["id"].values

        # Save
        save_submission(test_ids, predictions, Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation AUC ({final_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
