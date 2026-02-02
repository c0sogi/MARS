import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, AverageMeter, mapk
from library.dataset import get_dataloaders
from library.model import EfficientNetArcFace


def train_fn(dataloader, model, criterion, optimizer, device, epoch):
    """
    Executes one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass (returns ArcFace logits when labels are provided)
        logits = model(images, labels)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, device):
    """
    Evaluates the model on the validation set using MAP@5.
    Computes cosine similarity between validation embeddings and ArcFace class centers.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Get normalized class centers from the ArcFace head
    # shape: (num_classes, embedding_size)
    centers = F.normalize(model.head.weight, p=2, dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # labels are needed for metric calculation

            # Get embeddings (Forward with labels=None returns embeddings)
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute Cosine Similarity: (B, E) @ (C, E)^T -> (B, C)
            # This represents the cosine similarity between each image and each class center
            logits = torch.matmul(embeddings, centers.T)

            # Get top 5 predictions
            _, top_indices = logits.topk(Config.TOP_K, dim=1)

            all_preds.extend(top_indices.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate MAP@5
    # mapk expects list of ground truth scalars and list of predicted lists
    score = mapk(all_labels, all_preds, k=Config.TOP_K)
    return score


def inference_fn(dataloader, model, device, label_map):
    """
    Generates predictions for the test set.
    Uses Test-Time Augmentation (TTA) and saves to submission.csv.
    """
    model.eval()
    test_preds = []

    # Get normalized class centers
    centers = F.normalize(model.head.weight, p=2, dim=1)

    print("Starting Inference on Test Set...")

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # 1. Forward Pass (Original)
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 2. TTA: Horizontal Flip
            if Config.TTA:
                images_flip = torch.flip(images, [3])
                embeddings_flip = model(images_flip)
                embeddings_flip = F.normalize(embeddings_flip, p=2, dim=1)

                # Average embeddings
                embeddings = (embeddings + embeddings_flip) / 2.0
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # 3. Compute Similarity
            logits = torch.matmul(embeddings, centers.T)

            # 4. Get Top 5
            _, top_indices = logits.topk(Config.TOP_K, dim=1)

            test_preds.extend(top_indices.cpu().numpy())

    # Decode predictions using label_map
    # label_map maps hotel_id -> label_idx
    # We need label_idx -> hotel_id
    idx_to_hotel = {v: k for k, v in label_map.items()}

    final_submission = []

    # Get list of image names from the dataset dataframe
    # Accessing the underlying dataset from the dataloader
    image_names = dataloader.dataset.df["image"].values

    for img_name, pred_indices in zip(image_names, test_preds):
        # Map indices to hotel_ids
        pred_hotel_ids = [str(idx_to_hotel[idx]) for idx in pred_indices]
        prediction_string = " ".join(pred_hotel_ids)
        final_submission.append({"image": img_name, "hotel_id": prediction_string})

    # Create DataFrame
    submission_df = pd.DataFrame(final_submission)

    # Save
    submission_path = Config.SUBMISSION_FILE
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())


def run_training():
    """
    Main execution function.
    """
    seed_everything(Config.SEED)

    # 1. Data Loaders
    print("Loading Data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    # Load Label Encoder for inference mapping
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.parquet")
    if os.path.exists(encoder_path):
        label_df = pd.read_parquet(encoder_path)
        label_map = dict(zip(label_df["hotel_id"], label_df["label_idx"]))
    else:
        raise FileNotFoundError(
            "Label encoder not found. It should have been created by get_dataloaders."
        )

    # 2. Model
    print(f"Initializing Model: {Config.MODEL_NAME} with {num_classes} classes...")
    device = torch.device(Config.DEVICE)
    model = EfficientNetArcFace(n_classes=num_classes).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # 4. Loss
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_map = 0.0
    patience = 3  # Early stopping patience
    patience_counter = 0

    print(f"Starting Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch)

        # Eval
        val_map = eval_fn(val_loader, model, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MAP@5: {val_map:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Checkpoint & Early Stopping
        if val_map > best_map:
            best_map = val_map
            print(f"New Best Score! Saving model to {Config.MODEL_PATH}")
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    # 6. Inference
    print("\nTraining Complete. Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    inference_fn(test_loader, model, device, label_map)
