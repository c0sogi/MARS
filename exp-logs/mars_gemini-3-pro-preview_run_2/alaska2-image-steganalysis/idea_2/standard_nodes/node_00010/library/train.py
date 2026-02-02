import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, AverageMeter, alaska_weighted_auc
from library.dataset import AlaskaDataset, get_transforms
from library.model import StegoNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    Handles the paired (Cover, Stego) batch structure by flattening it
    into a single batch for the forward pass.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images).view(-1)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Weighted AUC.
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images).view(-1)
            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Collect logits and targets for AUC calculation
            # We pass logits directly; rank-based metrics like AUC handle this fine,
            # or we can sigmoid them. alaska_weighted_auc uses roc_curve which accepts logits.
            all_preds.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Weighted AUC
    auc = alaska_weighted_auc(all_targets, all_preds)

    return losses.avg, auc


def inference(model, loader, device):
    """
    Generates predictions for the test set using 5-View TTA.
    Views: Original, Horizontal Flip, Vertical Flip, Rot90, Rot270.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            # images: (B, C, H, W)

            # Create 5 views
            # 1. Original
            x = images
            # 2. Horizontal Flip (W is dim 3)
            x_h = torch.flip(x, [3])
            # 3. Vertical Flip (H is dim 2)
            x_v = torch.flip(x, [2])
            # 4. Rotate 90 deg (k=1)
            x_r90 = torch.rot90(x, 1, [2, 3])
            # 5. Rotate 270 deg (k=3)
            x_r270 = torch.rot90(x, 3, [2, 3])

            # Stack along batch dimension: (B*5, C, H, W)
            # Order: [Batch_Orig, Batch_H, Batch_V, Batch_R90, Batch_R270]
            batch_stack = torch.cat([x, x_h, x_v, x_r90, x_r270], dim=0)

            # Forward pass
            logits = model(batch_stack).view(-1)
            probs = torch.sigmoid(logits)

            # Split back into views to average per image
            # Split into 5 chunks of size B
            chunks = torch.chunk(probs, 5, dim=0)
            # Stack to (B, 5)
            probs_stacked = torch.stack(chunks, dim=1)

            # Average probability across views
            avg_probs = probs_stacked.mean(dim=1).cpu().numpy()

            for id_, score in zip(ids, avg_probs):
                results.append((id_, score))

    return results


def fit(epochs=Config.epochs, batch_size=Config.train_batch_size, debug=Config.debug):
    """
    Main training routine.
    Initializes datasets, model, optimizer, and runs the training loop.
    Performs Early Stopping and saves the best model.
    Generates submission file at the end.
    """
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    print(f"Starting training run. Device: {device}, Epochs: {epochs}, Debug: {debug}")

    # 2. Data Loading
    train_dataset = AlaskaDataset("train", transform=get_transforms("train"))
    val_dataset = AlaskaDataset("val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = StegoNet(
        backbone_name=Config.backbone_name,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
    )
    model = model.to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler: Cosine Annealing matching total epochs
    # Cite solution_lesson_node_00004: Avoid static config dependencies
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.min_lr
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience = 10
    patience_counter = 0
    best_model_path = os.path.join(Config.checkpoint_dir, "best_model.pth")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {curr_lr:.8f} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val Weighted AUC: {val_auc:.8f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.8f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    # Save last model for reference
    torch.save(
        model.state_dict(), os.path.join(Config.checkpoint_dir, "last_model.pth")
    )

    # 6. Submission Generation
    print("Generating submission for test set...")

    # Load Best Model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Test Loader
    test_dataset = AlaskaDataset("test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Inference
    predictions = inference(model, test_loader, device)

    # Save to CSV
    df_sub = pd.DataFrame(predictions, columns=["Id", "Label"])
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
