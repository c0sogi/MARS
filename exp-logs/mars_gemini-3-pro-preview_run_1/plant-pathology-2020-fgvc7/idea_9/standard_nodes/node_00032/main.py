import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import cv2

# Import from library
from library.config import Config
from library.utils import seed_everything, SWAHelper
from library.dataset import get_datasets
from library.model import AppleResNet34
from library.loss import get_weighted_loss
from library.engine import train_one_epoch, predict_tta, save_submission


def update_bn(loader, model, device):
    """
    Updates Batch Normalization running statistics by doing a forward pass
    on the training data.
    """
    model.train()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            model(images)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # We use use_full_data=False to ensure we have a validation set for metrics
    train_dataset, val_dataset, test_dataset = get_datasets(use_full_data=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = AppleResNet34(pretrained=Config.PRETRAINED).to(device)

    # 4. Loss & Optimizer
    # Load train metadata to calculate class weights
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    criterion = get_weighted_loss(train_df, device=device, load_cached_data=True)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR_CONVERGENCE)

    # Scheduler: Cosine Annealing for the first phase
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS_CONVERGENCE)

    # SWA Helper
    swa_helper = SWAHelper()

    # 5. Training Loop
    print("Starting Training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Hybrid Schedule Logic
        if epoch <= Config.EPOCHS_CONVERGENCE:
            # Phase 1: Convergence
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            # Phase 2: SWA
            # Manually set LR
            for param_group in optimizer.param_groups:
                param_group["lr"] = Config.LR_SWA
            current_lr = Config.LR_SWA

        print(f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.6f}")

        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch <= Config.EPOCHS_CONVERGENCE:
            scheduler.step()
        else:
            # Capture weights for SWA
            swa_helper.update(model)

    # 6. SWA Finalization
    print("Finalizing SWA Model...")
    avg_weights = swa_helper.get_averaged_weights()
    if avg_weights is not None:
        model.load_state_dict(avg_weights)
    else:
        print("Warning: No SWA weights captured. Using last model state.")

    # Update BN statistics
    print("Updating BN statistics...")
    update_bn(train_loader, model, device)

    # 7. Validation Assessment
    print("Performing Validation...")
    model.eval()
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            val_probs.append(probs.cpu().numpy())
            val_targets.append(labels.numpy())

    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)

    # Convert targets to one-hot for ROC AUC
    # val_targets is (N,) indices
    # We need (N, Num_Classes)
    val_targets_onehot = np.zeros((val_targets.size, Config.NUM_CLASSES))
    val_targets_onehot[np.arange(val_targets.size), val_targets] = 1

    try:
        metric = roc_auc_score(
            val_targets_onehot, val_probs, average="macro", multi_class="ovr"
        )
    except Exception as e:
        print(f"Error calculating metric: {e}")
        metric = 0.0

    print(f"Final Validation Metric: {metric}")

    # 8. Failure Analysis
    print("Performing Failure Analysis...")
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    errors = []
    widths = []
    heights = []
    intensities = []

    # Iterate through validation set to collect metadata and compute errors
    # Note: val_loader order matches val_meta_df because shuffle=False

    for idx, row in val_meta_df.iterrows():
        # Calculate Error: 1 - Probability of True Class
        true_class_idx = val_targets[idx]
        pred_prob = val_probs[idx, true_class_idx]
        error = 1.0 - pred_prob
        errors.append(error)

        # Load Image Stats
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Use cv2 to read
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            intensity = img.mean()
            widths.append(w)
            heights.append(h)
            intensities.append(intensity)
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "intensity": intensities}
    )

    # Compute Correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Error Correlations with Input Features:")
    print(correlations)

    # 9. Submission
    THRESHOLD = 0.9871488489626378
    if metric > THRESHOLD:
        print(f"Validation metric {metric} > {THRESHOLD}. Generating submission...")
        submission_df = predict_tta(model, test_loader, device)
        save_submission(submission_df)
    else:
        print(f"Validation metric {metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
