import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.model import get_model
from library.dataset import (
    BSONDataset,
    get_transforms,
    train_collate_fn,
    eval_collate_fn,
)


class CategoryMapper:
    """
    Handles mapping between raw category_id (large int) and class index (0..N-1).
    """

    def __init__(self):
        # Load category names to ensure we have the complete universe of categories
        df = pd.read_csv(Config.CATEGORY_NAMES)
        # Sort to ensure deterministic mapping across runs
        self.categories = sorted(df["category_id"].unique().tolist())
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx2cat = {i: cat for i, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)

    def to_idx(self, category_ids):
        """Converts raw category_ids to class indices."""
        if isinstance(category_ids, torch.Tensor):
            category_ids = category_ids.tolist()
        # Use -1 for unknown/padding if necessary, though shouldn't happen in valid train data
        return torch.tensor(
            [self.cat2idx.get(c, -1) for c in category_ids], dtype=torch.long
        )

    def to_cat(self, indices):
        """Converts class indices back to raw category_ids."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return [self.idx2cat.get(i, -1) for i in indices]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        True  # Enable benchmark for speed on consistent input sizes
    )


def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler, loss_fn, device, mapper
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        # Map raw category IDs to 0-indexed labels
        targets = mapper.to_idx(labels).to(device)

        optimizer.zero_grad()

        # Automatic Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, targets)

        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step Scheduler (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Metrics tracking
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += batch_size

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def validate(model, dataloader, loss_fn, device, mapper):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for flat_images, labels, product_ids, counts in dataloader:
            flat_images = flat_images.to(device)
            targets = mapper.to_idx(labels).to(device)

            with autocast():
                # Forward pass on all images
                logits = model(flat_images)
                probs = torch.softmax(logits, dim=1)

                # Late Fusion: Aggregate probabilities per product
                # Split the flat probability tensor back into per-product chunks
                probs_split = torch.split(probs, counts.tolist())

                # Average probabilities across images for each product
                product_probs = torch.stack([p.mean(dim=0) for p in probs_split])

                # Calculate NLL Loss on the aggregated probabilities
                # Add epsilon for numerical stability
                log_probs = torch.log(product_probs + 1e-9)
                loss = nn.NLLLoss()(log_probs, targets)

            total_loss += loss.item() * targets.size(0)

            _, predicted = product_probs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def predict(model, dataloader, device, mapper):
    model.eval()
    results = []

    with torch.no_grad():
        for flat_images, labels, product_ids, counts in dataloader:
            flat_images = flat_images.to(device)

            with autocast():
                logits = model(flat_images)
                probs = torch.softmax(logits, dim=1)

                # Late Fusion
                probs_split = torch.split(probs, counts.tolist())
                product_probs = torch.stack([p.mean(dim=0) for p in probs_split])

            _, predicted_indices = product_probs.max(1)

            # Map indices back to original category_ids
            predicted_cats = mapper.to_cat(predicted_indices)
            pids = product_ids.tolist()

            for pid, cat in zip(pids, predicted_cats):
                results.append({"_id": pid, "category_id": cat})

    return pd.DataFrame(results)


def run(debug=Config.DEBUG, epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main execution function.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing Engine (Debug={debug}, Device={device})")

    # 1. Setup Data
    mapper = CategoryMapper()

    train_dataset = BSONDataset(
        Config.TRAIN_META,
        mode="train",
        transform=get_transforms("train", Config.IMG_SIZE),
        debug=debug,
    )
    val_dataset = BSONDataset(
        Config.VAL_META,
        mode="val",
        transform=get_transforms("val", Config.IMG_SIZE),
        debug=debug,
    )
    test_dataset = BSONDataset(
        Config.TEST_META,
        mode="test",
        transform=get_transforms("test", Config.IMG_SIZE),
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=eval_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=eval_collate_fn,
        pin_memory=True,
    )

    # 2. Setup Model & Training Components
    model = get_model(num_classes=mapper.num_classes, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    scaler = GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # 3. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, device, mapper
        )

        val_loss, val_acc = validate(model, val_loader, loss_fn, device, mapper)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.0f}s")
        print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")
        print(f"Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.6f}")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"-> New best model saved! Acc: {best_acc:.6f}")

    # 4. Inference
    print("\nLoading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model.")

    print("Generating predictions on Test set...")
    df_submission = predict(model, test_loader, device, mapper)

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
