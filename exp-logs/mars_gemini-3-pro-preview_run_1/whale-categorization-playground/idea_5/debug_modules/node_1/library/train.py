import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, AverageMeter, map_at_5, save_checkpoint
from library.dataset import get_loaders
from library.model import WhaleArcFaceModel


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over training data
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # When labels are provided, WhaleArcFaceModel applies the ArcFace margin penalty
        logits = model(images, labels)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, device):
    """
    Evaluates the model on the validation set using TTA (Horizontal Flip).
    Returns MAP@5 score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA: Original Image
            logits_orig = model(
                images, labels=None
            )  # labels=None returns raw cosine * s

            # TTA: Flipped Image
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            # Average Logits
            logits_avg = (logits_orig + logits_flip) / 2.0

            # Get Top 5 predictions
            # logits_avg shape: (Batch, Num_Classes)
            _, top_indices = torch.topk(logits_avg, k=5, dim=1)

            all_preds.extend(top_indices.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Calculate MAP@5
    score = map_at_5(all_preds, all_targets)
    return score


def inference(test_loader, model, device, class_names):
    """
    Generates predictions for the test set using TTA and saves to submission file.
    """
    print("Starting Inference on Test Set...")
    model.eval()

    results = []

    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)

            # TTA: Original Image
            logits_orig = model(images, labels=None)

            # TTA: Flipped Image
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            # Average Logits
            logits_avg = (logits_orig + logits_flip) / 2.0

            # Get Top 5 predictions
            _, top_indices = torch.topk(logits_avg, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()

            for img_name, indices in zip(image_names, top_indices):
                # Convert indices to class strings
                pred_strings = [class_names[idx] for idx in indices]
                pred_str = " ".join(pred_strings)
                results.append({"Image": img_name, "Id": pred_str})

    # Create DataFrame and save
    df_sub = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run():
    """
    Main execution function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")
    print(f"Artifacts will be saved to: {Config.WORKING_DIR}")

    # 2. Data Loaders
    print("Loading Data...")
    train_loader, val_loader, test_loader, class_names = get_loaders(
        load_cached_data=True
    )
    print(f"Classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    print("Initializing Model...")
    model = WhaleArcFaceModel(
        num_classes=len(class_names),
        backbone_name=Config.BACKBONE,
        embedding_dim=Config.EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        s=Config.ARCFACE_SCALE,
        m=Config.ARCFACE_MARGIN,
        pretrained=True,
    )
    model = model.to(device)

    # 4. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_map5 = 0.0

    print("Starting Training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_map5 = validate(val_loader, model, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAP@5: {val_map5:.10f}"
        )

        # Save Checkpoint
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            print(f"New Best MAP@5: {best_map5:.10f}")

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_map5": best_map5,
                "optimizer": optimizer.state_dict(),
                "class_names": class_names,
            },
            is_best=is_best,
            filename=f"checkpoint_epoch_{epoch}.pth.tar",
        )

    print(f"Training Complete. Best MAP@5: {best_map5:.10f}")

    # 6. Inference
    # Load best model
    print("Loading best model for inference...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    inference(test_loader, model, device, class_names)


if __name__ == "__main__":
    run()
