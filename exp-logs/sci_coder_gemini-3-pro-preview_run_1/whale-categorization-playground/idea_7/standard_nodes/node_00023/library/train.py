import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, AverageMeter, map5_metric, WhaleLabelEncoder
from library.dataset import WhaleDataset, get_transforms
from library.model import WhaleDenseNet


def get_logits_inference(model, images, device, tta=False):
    """
    Computes logits for inference/validation using the ArcFace weights.
    Applies Test-Time Augmentation (TTA) if specified.

    Args:
        model: The WhaleDenseNet model.
        images: Batch of images.
        device: Torch device.
        tta: Boolean, whether to apply horizontal flip TTA.

    Returns:
        logits: Cosine similarity logits scaled by ArcFace scale (s).
    """
    # Forward pass 1 (Original)
    # When labels=None, model returns embeddings
    features = model(images)

    if tta:
        # Horizontal Flip TTA
        images_flipped = torch.flip(images, dims=[3])
        features_flipped = model(images_flipped)
        # Average embeddings
        features = (features + features_flipped) / 2.0

    # Normalize features (L2)
    features_norm = F.normalize(features, p=2, dim=1)

    # Normalize ArcFace weights (L2)
    # model.arcface.weight shape: (num_classes, embedding_size)
    weights_norm = F.normalize(model.arcface.weight, p=2, dim=1)

    # Compute Cosine Similarity * Scale
    # (B, E) @ (C, E).T -> (B, C)
    logits = torch.mm(features_norm, weights_norm.t()) * model.arcface.s

    return logits


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Runs one epoch of training.
    """
    model.train()

    losses = AverageMeter()
    top5 = AverageMeter()

    for i, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass (Training: returns ArcFace logits with margin penalty)
        output = model(images, labels)
        loss = criterion(output, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        # Note: 'output' here contains margin penalties, so accuracy is a proxy
        acc5_score = map5_metric(torch.topk(output, 5)[1], labels)

        losses.update(loss.item(), images.size(0))
        top5.update(acc5_score, images.size(0))

    return losses.avg, top5.avg


def validate(val_loader, model, device):
    """
    Evaluates the model on the validation set using MAP@5.
    Uses TTA and pure cosine similarity (no margin).
    """
    model.eval()

    map5 = AverageMeter()

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get logits using inference logic (no margin, cosine similarity)
            logits = get_logits_inference(model, images, device, tta=Config.TTA_FLIP)

            # Predictions
            preds = torch.topk(logits, 5)[1]

            # Metric
            score = map5_metric(preds, labels)
            map5.update(score, images.size(0))

    return map5.avg


def train_model(seed):
    """
    Trains a single model instance with the given seed using the
    Multi-Stage Progressive Training pipeline.
    """
    print(f"Starting training for Seed {seed}")
    seed_everything(seed)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # Setup Data (Initial)
    # -------------------------------------------------------------------------
    # Load Train Data (Stage 1)
    # This also fits/loads the label encoder
    train_dataset_stage1 = WhaleDataset(
        Config.TRAIN_CSV,
        transform=get_transforms("train", Config.STAGE_1_IMG_SIZE),
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    label_encoder = train_dataset_stage1.label_encoder
    num_classes = label_encoder.num_classes()

    val_dataset_stage1 = WhaleDataset(
        Config.VAL_CSV,
        label_encoder=label_encoder,
        transform=get_transforms("val", Config.STAGE_1_IMG_SIZE),
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset_stage1,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset_stage1,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Setup Model
    # -------------------------------------------------------------------------
    model = WhaleDenseNet(
        num_classes=num_classes,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=Config.PRETRAINED,
        dropout_rate=Config.DROPOUT_RATE,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # Stage 1: Coarse Tuning
    # -------------------------------------------------------------------------
    print(
        f"Stage 1: Training at {Config.STAGE_1_IMG_SIZE}x{Config.STAGE_1_IMG_SIZE} for {Config.STAGE_1_EPOCHS} epochs."
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.STAGE_1_LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.STAGE_1_EPOCHS
    )

    best_map5 = 0.0
    best_stage1_path = os.path.join(
        Config.WORKING_DIR, f"model_seed_{seed}_best_stage1.pth"
    )

    for epoch in range(1, Config.STAGE_1_EPOCHS + 1):
        train_loss, train_map5 = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )
        val_map5 = validate(val_loader, model, device)
        scheduler.step()

        print(
            f"Epoch {epoch} (Stage 1) - Train Loss: {train_loss:.6f} - Train MAP@5: {train_map5:.6f} - Val MAP@5: {val_map5}"
        )

        if val_map5 > best_map5:
            best_map5 = val_map5
            torch.save(model.state_dict(), best_stage1_path)

    # Load best stage 1 model before stage 2
    if os.path.exists(best_stage1_path):
        print("Loading best Stage 1 model for Stage 2 initialization...")
        model.load_state_dict(torch.load(best_stage1_path, map_location=device))

    # -------------------------------------------------------------------------
    # Stage 2: Fine Tuning
    # -------------------------------------------------------------------------
    print(
        f"Stage 2: Training at {Config.STAGE_2_IMG_SIZE}x{Config.STAGE_2_IMG_SIZE} for {Config.STAGE_2_EPOCHS} epochs."
    )

    # Re-initialize Datasets and Loaders with new resolution
    train_dataset_stage2 = WhaleDataset(
        Config.TRAIN_CSV,
        label_encoder=label_encoder,
        transform=get_transforms("train", Config.STAGE_2_IMG_SIZE),
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    val_dataset_stage2 = WhaleDataset(
        Config.VAL_CSV,
        label_encoder=label_encoder,
        transform=get_transforms("val", Config.STAGE_2_IMG_SIZE),
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset_stage2,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset_stage2,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Re-initialize Optimizer with lower LR
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.STAGE_2_LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.STAGE_2_EPOCHS
    )

    final_model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

    for epoch in range(1, Config.STAGE_2_EPOCHS + 1):
        train_loss, train_map5 = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )
        val_map5 = validate(val_loader, model, device)
        scheduler.step()

        print(
            f"Epoch {epoch} (Stage 2) - Train Loss: {train_loss:.6f} - Train MAP@5: {train_map5:.6f} - Val MAP@5: {val_map5}"
        )

        if val_map5 > best_map5:
            best_map5 = val_map5
            torch.save(model.state_dict(), final_model_path)

    # If Stage 2 didn't improve, ensure we have the best model saved as the final output
    if not os.path.exists(final_model_path):
        if os.path.exists(best_stage1_path):
            print("Stage 2 did not improve. Using best Stage 1 model.")
            model.load_state_dict(torch.load(best_stage1_path, map_location=device))
            torch.save(model.state_dict(), final_model_path)
        else:
            # Should not happen unless training failed completely
            torch.save(model.state_dict(), final_model_path)

    print(f"Finished training for Seed {seed}. Best Val MAP@5: {best_map5}")


def run():
    """
    Main execution function.
    1. Trains the ensemble of models.
    2. Performs inference on the test set using the ensemble.
    3. Generates the submission file.
    """
    # 1. Train Ensemble
    for seed in Config.ENSEMBLE_SEEDS:
        train_model(seed)

    # 2. Ensemble Inference
    print("Starting Ensemble Inference...")
    device = torch.device(Config.DEVICE)

    # Load Label Encoder (cached from training)
    label_encoder = WhaleLabelEncoder()
    # Pass empty list for IDs, relying on load_cached_data=True to pick up the parquet file
    label_encoder.fit([], load_cached_data=True)

    # Setup Test Dataset (Stage 2 Resolution)
    test_dataset = WhaleDataset(
        Config.TEST_CSV,
        label_encoder=None,  # No labels
        transform=get_transforms("val", Config.STAGE_2_IMG_SIZE),
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulate logits from all models
    avg_logits = None

    for seed in Config.ENSEMBLE_SEEDS:
        print(f"Inference with model seed {seed}...")

        # Initialize Model
        model = WhaleDenseNet(
            num_classes=label_encoder.num_classes(),
            embedding_size=Config.EMBEDDING_SIZE,
            pretrained=False,
            dropout_rate=Config.DROPOUT_RATE,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )
        model.to(device)
        model.eval()

        # Load Checkpoint
        ckpt_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found for seed {seed}. Skipping.")
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Generate Logits
        model_logits = []

        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(device)
                logits = get_logits_inference(
                    model, images, device, tta=Config.TTA_FLIP
                )
                model_logits.append(logits.cpu())

        model_logits = torch.cat(model_logits, dim=0)

        if avg_logits is None:
            avg_logits = model_logits
        else:
            avg_logits += model_logits

    # Average logits
    avg_logits /= len(Config.ENSEMBLE_SEEDS)

    # 3. Generate Submission
    print("Generating submission file...")

    # Get Top 5 indices
    top5_indices = torch.topk(avg_logits, Config.TOP_K, dim=1)[1].numpy()

    # Decode to strings
    image_names = test_dataset.df["Image"].values
    submission_rows = []

    for i, indices in enumerate(top5_indices):
        pred_labels = label_encoder.inverse_transform(indices)
        pred_string = " ".join(pred_labels)
        submission_rows.append({"Image": image_names[i], "Id": pred_string})

    # Save
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
