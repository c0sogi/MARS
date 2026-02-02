import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from library.config import Config
from library.dataset import create_dataloaders
from library.model import HierarchicalAttentionResNet
from library.utils import calculate_accuracy, load_category_hierarchy


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random is imported in utils and seeded there, but good practice to enforce here if needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_inverse_category_mapping():
    """
    Creates a mapping from model output index (l3_idx) to original category_id.
    """
    df_hierarchy = load_category_hierarchy(load_cached_data=True)
    # The dataframe is indexed by category_id and has l3_idx column.
    # We need l3_idx -> category_id
    # Ensure l3_idx is integer
    df_hierarchy["l3_idx"] = df_hierarchy["l3_idx"].astype(int)

    # Create dictionary
    idx_to_cat = pd.Series(
        df_hierarchy.index.values, index=df_hierarchy["l3_idx"]
    ).to_dict()
    return idx_to_cat


def train_one_epoch(model, loader, optimizer, scheduler, criterion_dict, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()

    running_loss = 0.0
    correct_l1 = 0
    correct_l2 = 0
    correct_l3 = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["images"].to(device, non_blocking=True)
        batch_index = batch["batch_index"].to(device, non_blocking=True)

        target_l1 = batch["l1_target"].to(device, non_blocking=True)
        target_l2 = batch["l2_target"].to(device, non_blocking=True)
        target_l3 = batch["l3_target"].to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, batch_index)

        # Compute Losses
        loss_l1 = criterion_dict["l1"](outputs["logits_l1"], target_l1)
        loss_l2 = criterion_dict["l2"](outputs["logits_l2"], target_l2)
        loss_l3 = criterion_dict["l3"](outputs["logits_l3"], target_l3)

        # Weighted Sum
        total_loss = (
            loss_l3 * Config.LOSS_WEIGHT_L3
            + loss_l2 * Config.LOSS_WEIGHT_L2
            + loss_l1 * Config.LOSS_WEIGHT_L1
        )

        # Backward
        total_loss.backward()

        # Optimizer Step
        optimizer.step()

        # Scheduler Step (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        # Metrics
        batch_size = target_l3.size(0)
        running_loss += total_loss.item() * batch_size
        total_samples += batch_size

        with torch.no_grad():
            correct_l1 += (outputs["logits_l1"].argmax(1) == target_l1).sum().item()
            correct_l2 += (outputs["logits_l2"].argmax(1) == target_l2).sum().item()
            correct_l3 += (outputs["logits_l3"].argmax(1) == target_l3).sum().item()

    avg_loss = running_loss / total_samples
    acc_l1 = correct_l1 / total_samples
    acc_l2 = correct_l2 / total_samples
    acc_l3 = correct_l3 / total_samples

    duration = time.time() - start_time

    print(
        f"Epoch {epoch+1} Train | Loss: {avg_loss:.6f} | L1 Acc: {acc_l1:.6f} | L2 Acc: {acc_l2:.6f} | L3 Acc: {acc_l3:.6f} | Time: {duration:.2f}s"
    )

    return avg_loss, acc_l3


def validate(model, loader, criterion_dict, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    running_loss = 0.0
    correct_l1 = 0
    correct_l2 = 0
    correct_l3 = 0
    total_samples = 0

    start_time = time.time()

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
            total_samples += batch_size

            correct_l1 += (outputs["logits_l1"].argmax(1) == target_l1).sum().item()
            correct_l2 += (outputs["logits_l2"].argmax(1) == target_l2).sum().item()
            correct_l3 += (outputs["logits_l3"].argmax(1) == target_l3).sum().item()

    avg_loss = running_loss / total_samples
    acc_l1 = correct_l1 / total_samples
    acc_l2 = correct_l2 / total_samples
    acc_l3 = correct_l3 / total_samples

    duration = time.time() - start_time

    print(
        f"Epoch Val   | Loss: {avg_loss:.6f} | L1 Acc: {acc_l1:.6f} | L2 Acc: {acc_l2:.6f} | L3 Acc: {acc_l3:.6f} | Time: {duration:.2f}s"
    )

    return avg_loss, acc_l3


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()

    idx_to_cat = get_inverse_category_mapping()

    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            images = batch["images"].to(device)
            batch_index = batch["batch_index"].to(device)
            sample_ids = batch["sample_ids"].cpu().numpy()

            outputs = model(images, batch_index)

            # Get predictions for L3 (Fine-grained)
            preds_l3 = outputs["logits_l3"].argmax(dim=1).cpu().numpy()

            for sid, pred_idx in zip(sample_ids, preds_l3):
                # Map index back to category_id
                cat_id = idx_to_cat.get(pred_idx, -1)  # -1 fallback
                results.append({"_id": sid, "category_id": cat_id})

    df_sub = pd.DataFrame(results)

    # Ensure integer types
    df_sub["_id"] = df_sub["_id"].astype(int)
    df_sub["category_id"] = df_sub["category_id"].astype(int)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function.
    """
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders()
    print(f"Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")

    # 2. Model
    print("Initializing Model...")
    model = HierarchicalAttentionResNet().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
        pct_start=0.3,
    )

    # 4. Loss Functions
    # Label Smoothing for L3 (Target)
    criterion_l3 = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    # Standard CE for auxiliary tasks
    criterion_l1 = nn.CrossEntropyLoss()
    criterion_l2 = nn.CrossEntropyLoss()

    criteria = {"l1": criterion_l1, "l2": criterion_l2, "l3": criterion_l3}

    # 5. Training Loop
    best_val_acc = 0.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criteria, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criteria, device)

        # Save Best Model
        if val_acc > best_val_acc:
            print(
                f"Validation L3 Accuracy improved from {best_val_acc:.6f} to {val_acc:.6f}. Saving checkpoint."
            )
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
        else:
            print(f"Validation L3 Accuracy did not improve (Best: {best_val_acc:.6f}).")

    print("\nTraining Complete.")

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    run_training()
