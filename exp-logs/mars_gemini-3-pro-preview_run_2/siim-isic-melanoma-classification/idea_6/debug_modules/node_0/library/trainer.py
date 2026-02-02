import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup

from library.utils import seed_everything, AverageMeter, weight_soup
from library.data_loader import get_dataloaders
from library.model import HierarchicalEfficientNet

# Configuration Constants
WORKING_DIR = "./working/idea_6"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion_primary, criterion_aux, device
):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Unpack batch
        images = batch["image"].to(device)
        meta = batch["meta"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # (B, 1)
        diagnosis = batch["diagnosis"].to(device)  # (B,)

        optimizer.zero_grad()

        # Forward pass
        primary_logits, aux_logits = model(images, meta)

        # Compute Losses
        loss_primary = criterion_primary(primary_logits, targets)
        loss_aux = criterion_aux(aux_logits, diagnosis)

        # Composite Loss: Primary + 0.1 * Aux
        loss = loss_primary + 0.1 * loss_aux

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion_primary, criterion_aux, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            diagnosis = batch["diagnosis"].to(device)

            # Forward pass
            primary_logits, aux_logits = model(images, meta)

            # Compute Loss
            loss_primary = criterion_primary(primary_logits, targets)
            loss_aux = criterion_aux(aux_logits, diagnosis)
            loss = loss_primary + 0.1 * loss_aux

            losses.update(loss.item(), images.size(0))

            # Store predictions for AUC calculation
            probs = torch.sigmoid(primary_logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case where only one class is present in batch/set
        auc = 0.5

    return losses.avg, auc


def fit_model(
    epochs=12,
    batch_size=32,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    load_cached_data=True,
):
    """
    Main training routine.
    """
    seed_everything(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Starting training on device: {device}")

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, num_diag_classes = get_dataloaders(
        batch_size=batch_size,
        image_size=384,
        num_workers=4,
        load_cached_data=load_cached_data,
    )

    # 2. Model Initialization
    print("Initializing Model...")
    # Determine number of metadata features from a sample batch
    sample_batch = next(iter(train_loader))
    n_meta_features = sample_batch["meta"].shape[1]

    model = HierarchicalEfficientNet(
        model_name="efficientnet_b3",
        pretrained=True,
        n_meta_features=n_meta_features,
        n_diagnosis_classes=num_diag_classes,
        num_classes=1,
    )
    model.to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Loss Functions
    # Positive class weight ~55.0 based on EDA (Imbalance 1:55)
    pos_weight = torch.tensor([55.0]).to(device)
    criterion_primary = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_aux = nn.CrossEntropyLoss()

    # Scheduler: Cosine with Warmup
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training Loop
    best_auc = 0.0
    checkpoint_history = []  # List of (auc, epoch, path)

    print(f"Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion_primary,
            criterion_aux,
            device,
        )

        val_loss, val_auc = validate_one_epoch(
            model, val_loader, criterion_primary, criterion_aux, device
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Save Checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        checkpoint_history.append((val_auc, epoch, ckpt_path))

    # 5. Checkpoint Averaging (Model Soup)
    print("Selecting top 3 checkpoints for averaging...")
    # Sort by AUC descending
    checkpoint_history.sort(key=lambda x: x[0], reverse=True)
    top_3_checkpoints = checkpoint_history[:3]

    print("Top 3 Epochs:")
    for auc, ep, path in top_3_checkpoints:
        print(f"  Epoch {ep}: AUC = {auc}")

    top_3_paths = [x[2] for x in top_3_checkpoints]

    # Create Model Soup
    print("Averaging weights...")
    averaged_weights = weight_soup(top_3_paths)

    # Save Best Averaged Model
    best_model_path = os.path.join(WORKING_DIR, "model_best.pth")
    torch.save(averaged_weights, best_model_path)
    print(f"Saved averaged model to {best_model_path}")

    return best_model_path
