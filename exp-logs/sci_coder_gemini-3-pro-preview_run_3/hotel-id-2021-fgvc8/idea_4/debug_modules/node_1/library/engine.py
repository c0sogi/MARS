import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, seed_everything
from library.model import HotelRecognitionModel
from library.dataset import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: returns ArcFace logits when labels are provided
        logits = model(images, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Performs one epoch of validation.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: returns ArcFace logits when labels are provided
            logits = model(images, labels)
            loss = criterion(logits, labels)

            loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def train_model():
    """
    Main function to train the model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Initialize Model
    model = HotelRecognitionModel()
    model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # Training Loop
    best_loss = float("inf")
    patience = 3
    counter = 0

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs.")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break


def generate_submission():
    """
    Main function to generate submission predictions.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data (Test Loader and Label Map)
    _, _, test_loader, idx_to_hotel = get_dataloaders(load_cached_data=True)

    # Load Model
    model = HotelRecognitionModel()
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 1. Extract Class Centers (Weights from ArcFace Head)
    # The weights in ArcMarginProduct are (out_features, in_features)
    # We normalize them to unit length for Cosine Similarity
    class_centers = model.arcface.weight.data
    class_centers = F.normalize(class_centers, p=2, dim=1)

    # 2. Extract Test Embeddings
    test_embeddings = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Original Forward
            emb_orig = model(images, labels=None)  # (B, Emb_Dim)

            # Test-Time Augmentation (Horizontal Flip)
            if Config.USE_TTA:
                images_flip = torch.flip(images, dims=[3])
                emb_flip = model(images_flip, labels=None)
                emb = (emb_orig + emb_flip) / 2.0
            else:
                emb = emb_orig

            # Normalize Embeddings
            emb = F.normalize(emb, p=2, dim=1)
            test_embeddings.append(emb)

    # Concatenate all batches
    test_embeddings = torch.cat(test_embeddings, dim=0)  # (N_Test, Emb_Dim)

    # 3. Compute Cosine Similarity
    # Similarity = Test_Embeddings @ Class_Centers.T
    # Shape: (N_Test, Num_Classes)
    # Ensure centers are on the same device
    class_centers = class_centers.to(device)
    similarity = torch.matmul(test_embeddings, class_centers.t())

    # 4. Rank and Select Top-K
    _, top_k_indices = torch.topk(similarity, k=Config.TOP_K, dim=1)
    top_k_indices = top_k_indices.cpu().numpy()

    # 5. Format Submission
    dataset = test_loader.dataset
    image_filenames = dataset.df["image"].values

    submission_rows = []
    for i, filename in enumerate(image_filenames):
        indices = top_k_indices[i]
        # Map indices back to hotel IDs
        hotel_ids = [str(idx_to_hotel[idx]) for idx in indices]
        prediction_str = " ".join(hotel_ids)
        submission_rows.append({"image": filename, "hotel_id": prediction_str})

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
