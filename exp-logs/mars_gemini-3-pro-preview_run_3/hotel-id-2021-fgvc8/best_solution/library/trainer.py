import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import (
    AverageMeter,
    mean_average_precision,
    save_checkpoint,
    seed_everything,
    load_checkpoint,
)
from library.dataset import HotelDataset, get_transforms, get_label_encoder
from library.model import HotelModel
from library.loss import ArcFaceLoss


def train_fn(train_loader, model, criterion, optimizer, scheduler, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    # Enable mixed precision for A100 optimization
    scaler = torch.cuda.amp.GradScaler()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # Forward pass with labels to apply ArcFace margin
            logits = model(images, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def eval_fn(val_loader, model, device):
    """
    Evaluates the model on the validation set using MAP@5.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get embeddings
            embeddings = model(images, labels=None)

            # Pass through head without labels to get cosine similarities (scaled)
            # This represents the confidence scores for each class
            logits = model.head(embeddings, labels=None)

            all_preds.append(logits.cpu())
            all_targets.append(labels.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute MAP@5
    map5 = mean_average_precision(predictions, targets, k=5)
    return map5


def generate_submission(test_loader, model, label_encoder, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print(f"Generating submission to {output_path}...")
    model.eval()

    image_ids = []
    preds_str = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # Get embeddings and logits
            embeddings = model(images, labels=None)
            logits = model.head(embeddings, labels=None)

            # Get top 5 indices
            _, topk_indices = logits.topk(5, dim=1, largest=True, sorted=True)
            topk_indices = topk_indices.cpu().numpy()

            image_ids.extend(filenames)

            # Decode labels
            for indices in topk_indices:
                decoded_labels = label_encoder.inverse_transform(indices)
                # Format as space-delimited string
                pred_string = " ".join(map(str, decoded_labels))
                preds_str.append(pred_string)

    # Create DataFrame and save
    submission_df = pd.DataFrame({"image": image_ids, "hotel_id": preds_str})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved. Shape: {submission_df.shape}")


def run_training():
    """
    Main execution function for training, validation, and submission generation.
    """
    seed_everything(Config.seed)

    # -------------------------
    # Data Preparation
    # -------------------------
    print("Initializing Data...")

    # Label Encoder
    label_encoder = get_label_encoder(
        Config.train_csv, Config.working_dir, load_cached_data=True
    )

    # Datasets
    train_dataset = HotelDataset(
        Config.train_csv,
        Config.input_dir,
        label_encoder=label_encoder,
        transform=get_transforms(Config.image_size, mode="train"),
        debug=Config.debug,
    )

    val_dataset = HotelDataset(
        Config.val_csv,
        Config.input_dir,
        label_encoder=label_encoder,
        transform=get_transforms(Config.image_size, mode="val"),
        debug=Config.debug,
    )

    test_dataset = HotelDataset(
        Config.test_csv,
        Config.input_dir,
        is_test=True,
        transform=get_transforms(Config.image_size, mode="test"),
        debug=Config.debug,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Model Initialization
    # -------------------------
    print(f"Initializing Model: {Config.model_name}")
    model = HotelModel(
        num_classes=Config.num_classes,
        model_name=Config.model_name,
        embedding_size=Config.embedding_size,
        scale=Config.scale,
        margin=Config.margin,
        k_subcenters=Config.k_subcenters,
        pretrained=Config.pretrained,
    )
    model.to(Config.device)

    # -------------------------
    # Optimization Setup
    # -------------------------
    criterion = ArcFaceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler: Linear Warmup + Cosine Decay
    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = len(train_loader) * Config.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # -------------------------
    # Training Loop
    # -------------------------
    best_map = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(1, Config.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, Config.device, epoch
        )

        # Validate
        val_map = eval_fn(val_loader, model, Config.device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAP@5: {val_map:.10f}"
        )

        # Checkpointing
        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            patience_counter = 0
            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_map,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filepath=os.path.join(Config.working_dir, "last_checkpoint.pth"),
                best_filepath=Config.model_path,
            )
            print(f"  >>> New Best MAP@5: {best_map:.10f} (Saved)")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{Config.patience}")

        # Early Stopping
        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    # -------------------------
    # Submission Generation
    # -------------------------
    print("\nTraining complete. Loading best model for submission...")

    # Load best weights
    _, best_score = load_checkpoint(model, Config.model_path, device=Config.device)
    print(f"Loaded model with Best MAP@5: {best_score:.10f}")

    generate_submission(
        test_loader, model, label_encoder, Config.device, Config.submission_path
    )
