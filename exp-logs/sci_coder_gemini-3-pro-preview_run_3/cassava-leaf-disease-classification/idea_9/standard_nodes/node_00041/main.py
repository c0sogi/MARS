import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

from library.config import Config
from library.utils import seed_everything
from library.data import CassavaDataset, get_transforms
from library.model import CassavaClassifier
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import generate_submission, inference_fn


def run_training(model_name, save_filename):
    """
    Trains a single model and saves the best weights.
    """
    print(f"\nStarting training for {model_name}...")

    # Data Preparation
    train_dataset = CassavaDataset(
        Config.train_metadata, transform=get_transforms("train"), is_train=True
    )
    val_dataset = CassavaDataset(
        Config.val_metadata, transform=get_transforms("valid"), is_train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model Setup
    device = torch.device(Config.device)
    model = CassavaClassifier(model_name, pretrained=True)
    model.to(device)

    # Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.min_lr)
    scaler = GradScaler()

    best_acc = 0.0
    save_path = os.path.join(Config.working_dir, save_filename)

    # Training Loop
    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, scaler, device
        )
        val_loss, val_acc = valid_one_epoch(epoch, model, val_loader, device)

        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  New best accuracy: {best_acc:.6f}. Saved to {save_path}")

    print(f"Finished training {model_name}. Best Accuracy: {best_acc:.6f}")

    # Clean up to free memory
    del (
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    )
    torch.cuda.empty_cache()

    return best_acc


def main():
    seed_everything(Config.seed)

    # 1. Train Model A (ViT)
    run_training(Config.model_a_name, "model_a_best.pth")

    # 2. Train Model B (BEiT)
    run_training(Config.model_b_name, "model_b_best.pth")

    # 3. Ensemble Validation
    print("\nRunning Ensemble Validation...")
    device = torch.device(Config.device)

    # Load Validation Data
    val_dataset = CassavaDataset(
        Config.val_metadata, transform=get_transforms("valid"), is_train=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Load Model A
    print("Loading Model A for inference...")
    model_a = CassavaClassifier(Config.model_a_name, pretrained=False)
    model_a.load_state_dict(
        torch.load(
            os.path.join(Config.working_dir, "model_a_best.pth"), map_location=device
        )
    )
    model_a.to(device)
    preds_a = inference_fn(model_a, val_loader, device)
    del model_a
    torch.cuda.empty_cache()

    # Load Model B
    print("Loading Model B for inference...")
    model_b = CassavaClassifier(Config.model_b_name, pretrained=False)
    model_b.load_state_dict(
        torch.load(
            os.path.join(Config.working_dir, "model_b_best.pth"), map_location=device
        )
    )
    model_b.to(device)
    preds_b = inference_fn(model_b, val_loader, device)
    del model_b
    torch.cuda.empty_cache()

    # Ensemble Predictions
    avg_probs = (preds_a + preds_b) / 2.0
    pred_labels = torch.argmax(avg_probs, dim=1).numpy()

    # Get True Labels
    true_labels = []
    for _, labels in val_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    # Compute Metric
    final_acc = (pred_labels == true_labels).mean()
    print(f"Final Validation Metric: {final_acc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude (1 - Probability of True Class)
    # avg_probs is [N, 5], true_labels is [N]
    probs_true_class = avg_probs[np.arange(len(true_labels)), true_labels].numpy()
    error_magnitude = 1.0 - probs_true_class

    # Calculate Input Features (Mean and Std of pixel intensity)
    # We iterate the loader again to compute stats on the normalized tensors
    img_means = []
    img_stds = []

    for images, _ in val_loader:
        # images: [B, C, H, W]
        # Calculate mean/std per image across C, H, W
        batch_means = images.mean(dim=[1, 2, 3]).numpy()
        batch_stds = images.std(dim=[1, 2, 3]).numpy()
        img_means.extend(batch_means)
        img_stds.extend(batch_stds)

    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Compute Correlations
    corr_mean = np.corrcoef(error_magnitude, img_means)[0, 1]
    corr_std = np.corrcoef(error_magnitude, img_stds)[0, 1]

    print(
        f"Correlation between Error Magnitude and Image Mean Intensity: {corr_mean:.6f}"
    )
    print(
        f"Correlation between Error Magnitude and Image Std Intensity: {corr_std:.6f}"
    )

    # 5. Submission
    threshold = 0.9049399198931909
    if final_acc > threshold:
        print(
            f"\nValidation metric {final_acc} > {threshold}. Generating submission..."
        )
        generate_submission(
            model_a_path=os.path.join(Config.working_dir, "model_a_best.pth"),
            model_b_path=os.path.join(Config.working_dir, "model_b_best.pth"),
        )
    else:
        print(f"\nValidation metric {final_acc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
