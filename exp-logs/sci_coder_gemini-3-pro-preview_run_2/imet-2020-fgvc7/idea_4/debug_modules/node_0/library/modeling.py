import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from library.config import Config
from library.utils import AverageMeter, get_logger, seed_everything
from library.losses import AsymmetricLoss, DistillationLoss
from library.dataset import get_dataloaders

logger = get_logger("modeling")


class ArtworkClassifier(nn.Module):
    """
    Wrapper for timm models to be used in the Artwork Attribute Labeling task.
    Initializes the backbone, adds global pooling (handled by timm), and attaches
    a linear classification head for multi-label prediction.
    """

    def __init__(self, model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(ArtworkClassifier, self).__init__()
        # timm.create_model with num_classes > 0 automatically adds the
        # appropriate global pooling and linear head.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        # Returns logits
        return self.model(x)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()

    # Progress bar only if not silent
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False, disable=True)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Check for soft targets (for distillation)
        soft_targets = None
        if "soft_target" in batch:
            soft_targets = batch["soft_target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)

            if isinstance(criterion, DistillationLoss) and soft_targets is not None:
                # DistillationLoss expects: student_logits, teacher_probs, hard_targets
                loss = criterion(logits, soft_targets, targets)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=losses.avg)
        pbar.update()

    pbar.close()
    return losses.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_logits = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        losses.update(loss.item(), images.size(0))
        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)

    return losses.avg, all_logits, all_targets


def optimize_threshold(logits, targets):
    """
    Finds the global threshold that maximizes Micro F1 score.
    """
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    best_thresh = 0.5
    best_f1 = 0.0

    # Search range
    thresholds = np.arange(0.2, 0.8, 0.05)

    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        score = f1_score(targets, preds, average="micro")
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh, best_f1


def train_model(
    model_name, checkpoint_path, epochs, use_distillation=False, soft_labels_path=None
):
    """
    Generic training function for both Teachers and Student.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Initialize Model
    model = ArtworkClassifier(model_name=model_name, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
    )

    # DataLoaders
    dataloaders = get_dataloaders(
        soft_labels_path=soft_labels_path if use_distillation else None,
        batch_size=Config.BATCH_SIZE,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR_MAX,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    # Loss Function
    if use_distillation:
        logger.info(f"Training {model_name} with DistillationLoss")
        criterion = DistillationLoss(alpha=Config.DISTILL_ALPHA)
    else:
        logger.info(f"Training {model_name} with AsymmetricLoss")
        criterion = AsymmetricLoss(
            gamma_neg=Config.ASL_GAMMA_NEG,
            gamma_pos=Config.ASL_GAMMA_POS,
            clip=Config.ASL_CLIP,
        )

    # Validation Criterion (always ASL or BCE for metric check)
    val_criterion = AsymmetricLoss()

    best_val_loss = float("inf")

    logger.info(f"Starting training for {model_name} over {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_loss, val_logits, val_targets = validate(
            model, val_loader, val_criterion, device
        )

        # Monitor Micro F1 at default threshold 0.5 for progress
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(val_targets, val_preds, average="micro")

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val MicroF1 (0.5): {val_f1:.6f}"
        )

        # Save Best Model (Based on Val Loss for calibration)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

    # Load best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


@torch.no_grad()
def generate_soft_labels(teacher_models, load_cached_data=True):
    """
    Generates soft labels using the ensemble of teacher models.
    Implements caching mechanism.
    """
    cache_path = Config.SOFT_LABELS_PATH

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached soft labels from {cache_path}")
        try:
            soft_labels = np.load(cache_path)
            return soft_labels
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Regenerating.")

    # 2. Compute from scratch
    logger.info("Generating soft labels from Teacher Ensemble...")
    device = Config.DEVICE

    # Get train loader without shuffle to align with metadata order
    # We reuse get_dataloaders but need a sequential sampler.
    # Since get_dataloaders returns shuffled train loader, we create a custom one here briefly.
    # To be safe and consistent with dataset.py logic:
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if Config.DEBUG:
        df_train = df_train.iloc[: Config.DEBUG_SUBSET_SIZE]

    from library.dataset import ArtworkDataset, get_transforms
    from torch.utils.data import DataLoader

    # Use validation transforms (deterministic) for soft label generation
    dataset = ArtworkDataset(
        df=df_train,
        input_dir=Config.INPUT_DIR,
        transforms=get_transforms("val"),  # No augmentation for teacher preds
        is_test=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Set teachers to eval
    for model in teacher_models:
        model.eval()
        model.to(device)

    all_probs = []

    for batch in tqdm(loader, desc="Generating Soft Labels"):
        images = batch["image"].to(device)

        batch_probs = []
        for model in teacher_models:
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

        # Average probabilities (Ensemble)
        avg_probs = np.mean(batch_probs, axis=0)
        all_probs.append(avg_probs)

    soft_labels = np.concatenate(all_probs, axis=0)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, soft_labels)
    logger.info(f"Saved soft labels to {cache_path}")

    return soft_labels


def predict_and_submit(model, threshold):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    logger.info("Generating test predictions...")
    device = Config.DEVICE
    dataloaders = get_dataloaders(batch_size=Config.VAL_BATCH_SIZE)
    test_loader = dataloaders["test"]

    model.eval()
    model.to(device)

    predictions = []
    ids = []

    for batch in tqdm(test_loader, desc="Test Inference"):
        images = batch["image"].to(device)
        batch_ids = batch["id"]

        with autocast():
            logits = model(images)
            probs = torch.sigmoid(logits)

        preds = (probs > threshold).int().cpu().numpy()

        for i, pred_row in enumerate(preds):
            # Convert binary vector to space-separated indices
            indices = np.where(pred_row == 1)[0]
            pred_str = " ".join(map(str, indices))
            predictions.append(pred_str)
            ids.append(batch_ids[i])

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "attribute_ids": predictions})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Main entry point for the Teacher-Student pipeline.
    """
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Stage 1: Train Teachers ---
    teachers = []

    # Teacher 1: ConvNeXt Small
    if os.path.exists(Config.TEACHER_1_CHECKPOINT):
        logger.info(f"Loading Teacher 1 from {Config.TEACHER_1_CHECKPOINT}")
        t1 = ArtworkClassifier(Config.TEACHER_1_MODEL_NAME, Config.NUM_CLASSES)
        t1.load_state_dict(
            torch.load(Config.TEACHER_1_CHECKPOINT, map_location=Config.DEVICE)
        )
    else:
        logger.info("Training Teacher 1 (ConvNeXt Small)...")
        t1 = train_model(
            Config.TEACHER_1_MODEL_NAME,
            Config.TEACHER_1_CHECKPOINT,
            epochs=Config.TEACHER_EPOCHS,
        )
    teachers.append(t1)

    # Teacher 2: Swin Base
    if os.path.exists(Config.TEACHER_2_CHECKPOINT):
        logger.info(f"Loading Teacher 2 from {Config.TEACHER_2_CHECKPOINT}")
        t2 = ArtworkClassifier(Config.TEACHER_2_MODEL_NAME, Config.NUM_CLASSES)
        t2.load_state_dict(
            torch.load(Config.TEACHER_2_CHECKPOINT, map_location=Config.DEVICE)
        )
    else:
        logger.info("Training Teacher 2 (Swin Base)...")
        t2 = train_model(
            Config.TEACHER_2_MODEL_NAME,
            Config.TEACHER_2_CHECKPOINT,
            epochs=Config.TEACHER_EPOCHS,
        )
    teachers.append(t2)

    # --- Stage 2: Generate Soft Labels ---
    # This function handles caching internally
    generate_soft_labels(teachers, load_cached_data=True)

    # Free up teacher memory
    del teachers, t1, t2
    torch.cuda.empty_cache()

    # --- Stage 3: Train Student with Distillation ---
    if os.path.exists(Config.STUDENT_CHECKPOINT):
        logger.info(f"Loading Student from {Config.STUDENT_CHECKPOINT}")
        student = ArtworkClassifier(Config.STUDENT_MODEL_NAME, Config.NUM_CLASSES)
        student.load_state_dict(
            torch.load(Config.STUDENT_CHECKPOINT, map_location=Config.DEVICE)
        )
        student.to(Config.DEVICE)
    else:
        logger.info("Training Student (ConvNeXt Base) with Distillation...")
        student = train_model(
            Config.STUDENT_MODEL_NAME,
            Config.STUDENT_CHECKPOINT,
            epochs=Config.STUDENT_EPOCHS,
            use_distillation=True,
            soft_labels_path=Config.SOFT_LABELS_PATH,
        )

    # --- Stage 4: Optimize Threshold ---
    logger.info("Optimizing Threshold on Validation Set...")
    dataloaders = get_dataloaders(batch_size=Config.VAL_BATCH_SIZE)
    val_criterion = AsymmetricLoss()
    _, val_logits, val_targets = validate(
        student, dataloaders["val"], val_criterion, Config.DEVICE
    )

    best_thresh, best_f1 = optimize_threshold(val_logits, val_targets)
    logger.info(
        f"Optimal Threshold: {best_thresh:.4f} with Val Micro F1: {best_f1:.6f}"
    )

    # --- Stage 5: Inference ---
    predict_and_submit(student, best_thresh)
    logger.info("Pipeline completed successfully.")
