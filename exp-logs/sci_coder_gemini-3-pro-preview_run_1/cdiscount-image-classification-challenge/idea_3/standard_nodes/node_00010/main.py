import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import time

# Ensure current directory is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import ProductDataset, collate_fn, get_transforms
from library.model import HierarchicalAttentionResNet
from library.utils import load_category_hierarchy
from torch.utils.data import DataLoader


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(Config.SEED)


# ==========================================
# Data Loading
# ==========================================
def create_custom_dataloaders():
    """
    Creates dataloaders with a specific optimization:
    - Train: Uses a subset (200k samples) for fast baseline training.
    - Val/Test: Uses the FULL dataset for valid metrics and submission.
    """
    # 1. Train Dataset (Subset)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200000

    train_transform = get_transforms(is_train=True)
    train_dataset = ProductDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        is_test=False,
        transform=train_transform,
    )

    # 2. Val & Test Datasets (Full)
    Config.DEBUG = False

    eval_transform = get_transforms(is_train=False)
    val_dataset = ProductDataset(
        metadata_path=Config.VAL_METADATA,
        bson_path=Config.TRAIN_BSON,
        is_test=False,
        transform=eval_transform,
    )

    test_dataset = ProductDataset(
        metadata_path=Config.TEST_METADATA,
        bson_path=Config.TEST_BSON,
        is_test=True,
        transform=eval_transform,
    )

    # 3. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    return train_loader, val_loader, test_loader


# ==========================================
# Training & Validation
# ==========================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion_dict, device):
    model.train()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        batch_index = batch["batch_index"].to(device, non_blocking=True)
        target_l1 = batch["l1_target"].to(device, non_blocking=True)
        target_l2 = batch["l2_target"].to(device, non_blocking=True)
        target_l3 = batch["l3_target"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, batch_index)

        loss_l1 = criterion_dict["l1"](outputs["logits_l1"], target_l1)
        loss_l2 = criterion_dict["l2"](outputs["logits_l2"], target_l2)
        loss_l3 = criterion_dict["l3"](outputs["logits_l3"], target_l3)

        total_loss = (
            loss_l3 * Config.LOSS_WEIGHT_L3
            + loss_l2 * Config.LOSS_WEIGHT_L2
            + loss_l1 * Config.LOSS_WEIGHT_L1
        )

        total_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        batch_size = target_l3.size(0)
        running_loss += total_loss.item() * batch_size
        correct_l3 += (outputs["logits_l3"].argmax(1) == target_l3).sum().item()
        total_samples += batch_size

    return running_loss / total_samples, correct_l3 / total_samples


def validate_and_analyze(model, loader, criterion_dict, device):
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    # Storage for Failure Analysis
    errors = []  # 1 if incorrect, 0 if correct
    num_imgs_list = []  # Number of images per product

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            batch_index = batch["batch_index"].to(device, non_blocking=True)
            target_l1 = batch["l1_target"].to(device, non_blocking=True)
            target_l2 = batch["l2_target"].to(device, non_blocking=True)
            target_l3 = batch["l3_target"].to(device, non_blocking=True)

            outputs = model(images, batch_index)

            loss_l1 = criterion_dict["l1"](outputs["logits_l1"], target_l1)
            loss_l2 = criterion_dict["l2"](outputs["logits_l2"], target_l2)
            loss_l3 = criterion_dict["l3"](outputs["logits_l3"], target_l3)

            total_loss = (
                loss_l3 * Config.LOSS_WEIGHT_L3
                + loss_l2 * Config.LOSS_WEIGHT_L2
                + loss_l1 * Config.LOSS_WEIGHT_L1
            )

            batch_size = target_l3.size(0)
            running_loss += total_loss.item() * batch_size

            preds = outputs["logits_l3"].argmax(1)
            is_correct = preds == target_l3
            correct_l3 += is_correct.sum().item()
            total_samples += batch_size

            # Failure Analysis Data Collection
            # Count images per product in this batch
            counts = torch.bincount(batch_index, minlength=batch_size)

            num_imgs_list.extend(counts.cpu().numpy())
            # Store Error (Inverse of correct)
            errors.extend((~is_correct).cpu().numpy().astype(int))

    avg_loss = running_loss / total_samples
    acc_l3 = correct_l3 / total_samples

    return avg_loss, acc_l3, np.array(errors), np.array(num_imgs_list)


# ==========================================
# Inference
# ==========================================
def generate_submission(model, loader, device, output_path):
    print("Generating submission...")
    model.eval()

    # Load hierarchy to map model output index back to category_id
    df_hierarchy = load_category_hierarchy(load_cached_data=True)
    df_hierarchy["l3_idx"] = df_hierarchy["l3_idx"].astype(int)
    idx_to_cat = pd.Series(
        df_hierarchy.index.values, index=df_hierarchy["l3_idx"]
    ).to_dict()

    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            batch_index = batch["batch_index"].to(device)
            sample_ids = batch["sample_ids"].cpu().numpy()

            outputs = model(images, batch_index)
            preds_l3 = outputs["logits_l3"].argmax(dim=1).cpu().numpy()

            for sid, pred_idx in zip(sample_ids, preds_l3):
                cat_id = idx_to_cat.get(pred_idx, -1)
                results.append({"_id": sid, "category_id": cat_id})

    df_sub = pd.DataFrame(results)
    df_sub["_id"] = df_sub["_id"].astype(int)
    df_sub["category_id"] = df_sub["category_id"].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# ==========================================
# Main
# ==========================================
def main():
    device = torch.device(Config.DEVICE)

    # 1. Prepare Data
    train_loader, val_loader, test_loader = create_custom_dataloaders()

    # 2. Initialize Model
    model = HierarchicalAttentionResNet().to(device)

    # 3. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
        pct_start=0.3,
    )

    criteria = {
        "l1": nn.CrossEntropyLoss(),
        "l2": nn.CrossEntropyLoss(),
        "l3": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
    }

    # 4. Training Loop
    best_val_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criteria, device
        )

        # Validate
        val_loss, val_acc, _, _ = validate_and_analyze(
            model, val_loader, criteria, device
        )

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    # 5. Final Reporting
    print(f"Final Validation Metric: {best_val_acc}")

    # 6. Failure Analysis
    # Reload best model for analysis
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    _, _, errors, num_imgs = validate_and_analyze(model, val_loader, criteria, device)

    if len(errors) > 0 and len(num_imgs) > 0:
        # Avoid division by zero in correlation if variance is 0
        if np.std(errors) > 0 and np.std(num_imgs) > 0:
            correlation = np.corrcoef(errors, num_imgs)[0, 1]
            print(
                f"Failure Analysis: Correlation between Error and NumImages: {correlation}"
            )
        else:
            print("Failure Analysis: Variance is zero, cannot compute correlation.")
    else:
        print("Failure Analysis: Insufficient data.")

    # 7. Submission
    THRESHOLD = 0.6306776302037904
    if best_val_acc > THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
