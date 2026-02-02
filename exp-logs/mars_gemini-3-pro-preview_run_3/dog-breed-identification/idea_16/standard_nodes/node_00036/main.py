import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import DogClassifier
from library.engine import train_head_only, train_one_epoch, validate
from library.soup import create_model_soup

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# We override specific Config attributes to ensure the run completes within the 45-minute limit
# while still validating the core architectural ideas.
Config.n_folds = 1  # Run only Fold 0 to save time
Config.finetune_epochs = 5  # Reduced epochs for fast verification
Config.soup_top_k = 3  # Average top 3 epochs
Config.debug = False  # Use full dataset for Fold 0 (approx 7k images), fits in time


def run():
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger("main", os.path.join(Config.working_dir, "run.log"))
    logger.info("Starting Hierarchical Stratified Ensemble - Fast Baseline")

    device = Config.device
    fold = 0  # Processing only the first fold

    # 2. Data Loading
    logger.info(f"Loading data for Fold {fold}...")
    train_loader, val_loader, classes = get_dataloaders(
        fold=fold, load_cached_data=True
    )

    # 3. Model Initialization
    logger.info(f"Initializing model: {Config.model_name}")
    model = DogClassifier(num_classes=len(classes), pretrained=True)
    model.to(device)

    # 4. Phase 1: Head Warmup
    # Train only the head with high LR to align with backbone features
    logger.info("Phase 1: Head Warmup (Frozen Backbone)")
    optimizer_head = torch.optim.AdamW(
        model.parameters(), lr=Config.warmup_lr, weight_decay=Config.weight_decay
    )
    train_head_only(model, train_loader, optimizer_head, device, epoch=0)

    # 5. Phase 2: Full Fine-tuning
    # Unfreeze backbone and train with lower LR and Cosine Annealing
    logger.info("Phase 2: Full Fine-tuning")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.finetune_lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.finetune_epochs, eta_min=Config.min_lr
    )

    checkpoint_dir = os.path.join(Config.working_dir, f"fold_{fold}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_metrics = []
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, Config.finetune_epochs + 1):
        # Train
        train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, _ = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Save Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        epoch_metrics.append({"epoch": epoch, "val_loss": val_loss, "path": ckpt_path})

    # 6. Manual Model Soup
    # Average weights of the best K epochs
    logger.info(f"Creating Model Soup from top {Config.soup_top_k} checkpoints...")
    epoch_metrics.sort(key=lambda x: x["val_loss"])
    top_k_paths = [x["path"] for x in epoch_metrics[: Config.soup_top_k]]

    soup_path = os.path.join(Config.working_dir, f"best_soup_fold_{fold}.pth")
    create_model_soup(top_k_paths, soup_path)

    # 7. Final Validation on Soup Model
    logger.info("Validating Soup Model...")
    model.load_state_dict(torch.load(soup_path, map_location=device))
    model.to(device)
    model.eval()

    final_val_loss, val_preds = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_loss}")

    # 8. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Retrieve Ground Truth
    y_true = []
    for i in range(len(val_loader.dataset)):
        _, label = val_loader.dataset[i]
        y_true.append(label.item())
    y_true = np.array(y_true)

    # Calculate Per-Sample Loss (Cross Entropy)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    sample_losses = []
    for i in range(len(y_true)):
        true_class = y_true[i]
        prob = val_preds_clipped[i, true_class]
        loss = -np.log(prob)
        sample_losses.append(loss)

    # Load Validation Metadata to get file paths
    # We need to match the validation set used by the loader
    df_folds = pd.read_parquet(os.path.join(Config.working_dir, "folds.parquet"))
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    # Extract Metadata Features (File Size, Aspect Ratio)
    file_sizes = []
    aspect_ratios = []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])
        try:
            # File Size
            f_size = os.path.getsize(full_path)

            # Aspect Ratio
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                ar = w / h if h > 0 else 0
            else:
                ar = 0
        except Exception:
            f_size = 0
            ar = 0

        file_sizes.append(f_size)
        aspect_ratios.append(ar)

    # Compute Correlations
    corr_size, _ = pearsonr(sample_losses, file_sizes)
    corr_ar, _ = pearsonr(sample_losses, aspect_ratios)

    print(f"Correlation (Loss vs File Size): {corr_size}")
    print(f"Correlation (Loss vs Aspect Ratio): {corr_ar}")

    # 9. Submission Generation
    # Only generate if metric is below threshold
    THRESHOLD = 0.14004325100369866

    if final_val_loss < THRESHOLD:
        logger.info("Metric below threshold. Generating submission with TTA...")
        test_loader, test_df = get_test_dataloader(load_cached_data=True)

        predictions = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                # TTA: Average of Original and Horizontal Flip
                # Original
                out1 = model(images)
                prob1 = torch.softmax(out1, dim=1)

                # Flip (Dim 3 is width)
                images_flipped = torch.flip(images, dims=[3])
                out2 = model(images_flipped)
                prob2 = torch.softmax(out2, dim=1)

                avg_prob = (prob1 + prob2) / 2.0
                predictions.append(avg_prob.cpu().numpy())

        final_preds = np.vstack(predictions)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(final_preds, columns=classes)
        sub_df.insert(0, "id", test_df["id"])

        sub_df.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(f"Metric {final_val_loss} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
