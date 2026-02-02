import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import CassavaDataset, get_transforms
from library.model import CassavaSwinModel
from library.engine import train_model, generate_submission


def analyze_failures_and_validate(val_loader, model, device):
    """
    Performs validation with TTA, computes the final metric, and runs failure analysis.
    """
    model.eval()
    y_true = []
    y_probs = []

    # Feature collectors for failure analysis
    feat_mean = []
    feat_std = []

    print("Running Validation with TTA and Failure Analysis...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass 1: Original
            out1 = model(images)

            # Forward pass 2: Horizontal Flip (TTA)
            if Config.USE_TTA:
                images_flip = torch.flip(images, dims=[3])
                out2 = model(images_flip)
                outputs = (out1 + out2) / 2.0
            else:
                outputs = out1

            probs = torch.softmax(outputs, dim=1)

            # Collect labels and probabilities
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

            # Calculate simple features from the batch of images for analysis
            # We use the normalized tensor values as proxies for brightness/contrast
            batch_mean = torch.mean(images, dim=[1, 2, 3]).cpu().numpy()
            batch_std = torch.std(images, dim=[1, 2, 3]).cpu().numpy()

            feat_mean.extend(batch_mean)
            feat_std.extend(batch_std)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    feat_mean = np.array(feat_mean)
    feat_std = np.array(feat_std)

    # Calculate Predictions
    y_pred = np.argmax(y_probs, axis=1)

    # Calculate and Print Final Metric
    acc = get_score(y_true, y_pred)
    print(f"Final Validation Metric: {acc}")

    # --- Failure Analysis ---
    # Error Magnitude: 1.0 - probability assigned to the correct class
    # Use advanced indexing to extract prob of ground truth label
    prob_correct = y_probs[np.arange(len(y_true)), y_true]
    error_mag = 1.0 - prob_correct

    # Calculate correlations
    # We check if error magnitude correlates with image brightness or contrast
    corr_mean, _ = pearsonr(error_mag, feat_mean)
    corr_std, _ = pearsonr(error_mag, feat_std)

    print("\nFailure Analysis - Feature Correlations with Error Magnitude:")
    print(f"Image Mean Intensity (Brightness): {corr_mean:.4f}")
    print(f"Image Std Deviation (Contrast):    {corr_std:.4f}")

    return acc


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_directories()

    # Override Config for fast baseline execution
    # Reducing epochs to 6 ensures we finish well within 2 hours on A100
    Config.EPOCHS = 6
    Config.T_MAX = Config.EPOCHS

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create Datasets
    train_dataset = CassavaDataset(
        df_train, transforms=get_transforms("train"), output_label=True
    )
    val_dataset = CassavaDataset(
        df_val, transforms=get_transforms("valid"), output_label=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = CassavaSwinModel()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eps=Config.EPS,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # 4. Training Loop
    print(f"Starting Training for {Config.EPOCHS} epochs...")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
    )

    # 5. Validation & Failure Analysis
    print("Loading best model for final analysis...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current model weights.")

    final_acc = analyze_failures_and_validate(val_loader, model, device)

    # 6. Submission Generation
    THRESHOLD = 0.9022696929238986

    if final_acc > THRESHOLD:
        print(
            f"Validation accuracy {final_acc} > {THRESHOLD}. Generating submission..."
        )

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = CassavaDataset(
            df_test, transforms=get_transforms("test"), output_label=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(model, test_loader, df_test, device)
    else:
        print(
            f"Validation accuracy {final_acc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
