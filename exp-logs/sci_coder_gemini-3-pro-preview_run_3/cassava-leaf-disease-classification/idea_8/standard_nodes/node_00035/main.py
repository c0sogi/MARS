import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import cv2
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Monkey-patch tqdm to suppress progress bars as per requirements
import tqdm


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = noop_tqdm

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_dataloaders
from library.models import get_model
from library.training import Trainer
from library.inference import generate_submission, predict_with_tta


def calculate_image_stats(file_paths, root_dir):
    """
    Calculates mean and std pixel values for a list of image paths.
    Used for failure analysis.
    """
    means = []
    stds = []

    for rel_path in file_paths:
        full_path = os.path.join(root_dir, rel_path)
        try:
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                means.append(np.mean(img))
                stds.append(np.std(img))
            else:
                means.append(0.0)
                stds.append(0.0)
        except Exception:
            means.append(0.0)
            stds.append(0.0)

    return np.array(means), np.array(stds)


def train_model(
    model_name, save_name, train_loader, val_loader, cfg, logger, lr, weight_decay
):
    """
    Orchestrates the training of a single model.
    """
    device = get_device()
    logger.info(f"Initializing {model_name}...")

    model = get_model(cfg, model_name)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)

    trainer = Trainer(model, train_loader, val_loader, cfg, logger)

    logger.info(f"Starting training for {model_name}...")
    best_acc = trainer.fit(optimizer, scheduler, epochs=cfg.EPOCHS, save_name=save_name)

    # Clean up to save memory
    del model
    del optimizer
    del trainer
    torch.cuda.empty_cache()

    return best_acc


def main():
    # 1. Configuration & Setup
    cfg = Config()

    # Optimize for Single Strong Learner
    # Cite solution_lesson_node_00015: ViT benefits from slightly longer training (10 epochs)
    cfg.EPOCHS = 10
    cfg.BATCH_SIZE = 32  # Standard for ViT-B/16 @ 384
    cfg.DEBUG = False

    seed_everything(cfg.SEED)
    device = get_device()

    # Setup Logger
    log_path = os.path.join(cfg.WORKING_DIR, "runfile.log")
    logger = get_logger(log_path)
    logger.info("Starting runfile execution...")

    # 2. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # 3. Train Model (ViT)
    model_save_name = "best_model.pth"
    train_model(
        model_name=cfg.MODEL_NAME,
        save_name=model_save_name,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        logger=logger,
        lr=cfg.MODEL_LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # 5. Validation
    logger.info("Starting Validation...")

    # Load Model
    path = os.path.join(cfg.WORKING_DIR, model_save_name)
    model = get_model(cfg, cfg.MODEL_NAME)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    # Get Predictions (using TTA for robustness)
    # Extract True Labels from Val Loader
    true_labels = []
    val_file_paths = []  # For failure analysis

    # Iterate once to get labels and paths
    val_df = pd.read_csv(cfg.VAL_METADATA)
    val_file_paths = val_df["file_path"].tolist()
    true_labels = val_df["label"].values

    # Model Probs
    probs, _ = predict_with_tta(model, val_loader, device, tta_steps=cfg.TTA_STEPS)

    # Predictions
    preds = np.argmax(probs, axis=1)

    # Calculate Metric
    final_acc = accuracy_score(true_labels, preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_acc}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate Error Magnitude: 1.0 - Probability assigned to the correct class
    rows = np.arange(len(true_labels))
    prob_correct = probs[rows, true_labels]
    error_magnitude = 1.0 - prob_correct

    # Calculate Image Stats
    mean_pixels, std_pixels = calculate_image_stats(val_file_paths, cfg.INPUT_ROOT)

    # Calculate Correlations
    corr_mean = np.corrcoef(error_magnitude, mean_pixels)[0, 1]
    corr_std = np.corrcoef(error_magnitude, std_pixels)[0, 1]

    print(f"Correlation (Error Magnitude vs Mean Pixel Intensity): {corr_mean}")
    print(f"Correlation (Error Magnitude vs Std Pixel Intensity): {corr_std}")

    # Clean up models
    del model
    torch.cuda.empty_cache()

    # 7. Submission
    THRESHOLD = 0.9022696929238986

    if final_acc > THRESHOLD:
        logger.info(
            f"Validation metric {final_acc} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(path)
    else:
        logger.info(
            f"Validation metric {final_acc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
