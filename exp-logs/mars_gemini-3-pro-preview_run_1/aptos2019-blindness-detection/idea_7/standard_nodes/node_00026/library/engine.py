import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.utils import seed_everything, save_checkpoint
from library.dataset import get_dataloaders
from library.model import RetinopathyModel, train_one_epoch, validate, inference


def run_training(
    epochs=10,
    batch_size=16,
    image_size=512,
    lr=1e-4,
    weight_decay=1e-2,
    seed=42,
    patience=4,
    output_dir="./working/idea_7",
):
    """
    Main execution function for the Diabetic Retinopathy classification task.
    Handles training, validation, early stopping, and inference.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        image_size (int): Resolution of input images.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        seed (int): Random seed for reproducibility.
        patience (int): Early stopping patience (epochs without improvement).
        output_dir (str): Directory to save checkpoints.
    """
    # 1. Setup
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting training run on device: {device}")

    # 2. Data
    # Load data using the provided library function
    # num_workers=8 is set based on system capabilities (12 vCPUs available)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, image_size=image_size, num_workers=8
    )

    # 3. Model
    print("Initializing ConvNeXt-Small Model with GeM Pooling...")
    model = RetinopathyModel(
        model_name="convnext_small.fb_in1k", pretrained=True, num_classes=4
    )
    model = model.to(device)

    # 4. Optimization
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop with Early Stopping
    best_score = -float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train Step
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, device)

        # Validation Step
        val_loss, val_score = validate(val_loader, model, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Log Metrics
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_score}")

        # Save Last Checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
            },
            is_best=False,
            checkpoint_dir=output_dir,
        )

        # Early Stopping Logic
        if val_score > best_score:
            print(
                f"Validation Score improved from {best_score} to {val_score}. Saving best model."
            )
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0

            # Save Best Checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                },
                is_best=True,
                checkpoint_dir=output_dir,
            )
        else:
            patience_counter += 1
            print(f"Score did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(
        f"\nTraining finished. Best Validation QWK: {best_score} at Epoch {best_epoch}"
    )

    # 6. Inference
    print("Starting Inference on Test Set (4-View TTA)...")

    # Load Best Model Weights
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded best model weights.")
    else:
        print("Warning: Best model not found. Using current model weights.")

    # Run Inference
    # The inference function in library.model handles TTA and decoding
    submission_df = inference(test_loader, model, device)

    # 7. Save Submission
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
