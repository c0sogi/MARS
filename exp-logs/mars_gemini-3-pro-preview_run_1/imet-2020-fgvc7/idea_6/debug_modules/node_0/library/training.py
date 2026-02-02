import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkModel
from library.utils import calculate_micro_f1, ModelEMA, seed_everything


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_mixup_cutmix(images, targets, config):
    """
    Applies Mixup or CutMix augmentation to the batch.
    Returns augmented images and mixed targets.
    """
    r = np.random.rand()
    if r > config.MIXUP_PROB:
        # No mixing, just apply label smoothing if needed later
        return images, targets, False

    batch_size = images.size(0)
    indices = torch.randperm(batch_size).to(images.device)

    # Decide between Mixup and CutMix
    # We can use a 50/50 split or another logic.
    # Here we assume equal probability if both are enabled,
    # but Config implies parameters. We'll choose randomly.
    use_cutmix = np.random.rand() < 0.5

    if use_cutmix:
        # CutMix
        lam = np.random.beta(config.CUTMIX_ALPHA, config.CUTMIX_ALPHA)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

        # Adjust lambda to exact area ratio
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
        )

        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        targets = lam * targets + (1 - lam) * targets[indices]
    else:
        # Mixup
        lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
        images = lam * images + (1 - lam) * images[indices]
        targets = lam * targets + (1 - lam) * targets[indices]

    return images, targets, True


def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, criterion, device, ema, epoch
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply Mixup/CutMix
        if Config.USE_MIXUP:
            images, targets, mixed = apply_mixup_cutmix(images, targets, Config)
            if not mixed:
                # Apply label smoothing manually if not mixed
                # y_smooth = y(1-eps) + 0.5*eps
                targets = (
                    targets * (1.0 - Config.LABEL_SMOOTHING)
                    + 0.5 * Config.LABEL_SMOOTHING
                )
        else:
            # Standard label smoothing
            targets = (
                targets * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update(model)

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss}")
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for predictions
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / dataset_size

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return epoch_loss, all_preds, all_targets


def optimize_thresholds(preds, targets):
    """
    Finds the optimal threshold for Micro F1 score.
    """
    best_threshold = 0.5
    best_f1 = 0.0

    # Grid search
    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
    )

    for thresh in thresholds:
        f1 = calculate_micro_f1(preds, targets, threshold=thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def generate_submission(model, device, threshold):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")

    # Setup Test Dataset and Loader
    test_dataset = ArtworkDataset(
        mode="test", load_cached_data=True, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    model.eval()
    submission_data = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            probs = probs.cpu().numpy()

            # Convert probabilities to labels
            for i, img_id in enumerate(ids):
                pred_indices = np.where(probs[i] > threshold)[0]
                pred_str = " ".join(map(str, pred_indices))
                submission_data.append({"id": img_id, "attribute_ids": pred_str})

    # Create DataFrame and Save
    df_sub = pd.DataFrame(submission_data)

    # Ensure columns are in correct order
    df_sub = df_sub[["id", "attribute_ids"]]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model(data_limit=None, num_epochs=None):
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing Datasets...")
    train_dataset = ArtworkDataset(
        mode="train",
        load_cached_data=True,
        transform=get_transforms("train"),
        data_limit=data_limit,
    )
    val_dataset = ArtworkDataset(
        mode="val",
        load_cached_data=True,
        transform=get_transforms("val"),
        data_limit=data_limit,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Model Setup
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = ArtworkModel(model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED)
    model.to(device)

    # Model EMA
    ema = None
    if Config.USE_EMA:
        print("Initializing Model EMA...")
        ema = ModelEMA(model, decay=Config.EMA_DECAY)

    # 3. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    epochs = num_epochs if num_epochs is not None else Config.EPOCHS

    # Cosine Annealing Scheduler
    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=Config.MIN_LR)

    # Loss Function
    # BCEWithLogitsLoss with positive weights
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = GradScaler()

    best_f1 = 0.0

    # 4. Training Loop
    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            device=device,
            ema=ema,
            epoch=epoch,
        )

        # Validate (Use EMA if available)
        eval_model = ema.ema if ema else model
        val_loss, val_preds, val_targets = validate(
            eval_model, val_loader, criterion, device
        )

        # Optimize Threshold for this epoch
        best_thresh_epoch, val_f1 = optimize_thresholds(val_preds, val_targets)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )
        print(f"Validation F1: {val_f1} (Threshold: {best_thresh_epoch})")

        # Save Best Model
        if val_f1 > best_f1:
            print(f"New Best F1! ({best_f1} -> {val_f1}). Saving model...")
            best_f1 = val_f1
            torch.save(eval_model.state_dict(), Config.MODEL_PATH)

    print(f"Training Complete. Best Validation F1: {best_f1}")

    # 5. Final Inference and Submission
    print("Loading best model for inference...")
    # Re-initialize model structure to load weights
    best_model = ArtworkModel(model_name=Config.MODEL_NAME, pretrained=False)
    best_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    best_model.to(device)

    # Re-calculate optimal threshold on validation set using best model
    # (Though we tracked it, it's safer to re-verify or just use the one from the best epoch)
    # To be precise, we'll run validation one last time to get the exact threshold for the saved weights
    print("Optimizing threshold on validation set with best model...")
    _, val_preds, val_targets = validate(best_model, val_loader, criterion, device)
    final_threshold, final_f1 = optimize_thresholds(val_preds, val_targets)
    print(f"Final Optimal Threshold: {final_threshold} (F1: {final_f1})")

    generate_submission(best_model, device, final_threshold)
