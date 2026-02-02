import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything
from library.model import HotelRecognitionModel
from library.dataset import get_dataloaders
from library.engine import train_one_epoch, generate_submission


def calculate_map5(model, dataloader, device):
    """
    Computes MAP@5 on the validation set.
    Returns:
        map5 (float): The Mean Average Precision @ 5.
        ap_scores (np.array): Average Precision score for each sample.
        targets (np.array): Ground truth indices for each sample.
    """
    model.eval()
    embeddings = []
    targets = []

    # 1. Extract Embeddings
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Forward pass with labels=None returns embeddings
            emb = model(images, labels=None)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)
            targets.append(labels)

    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0).to(device)

    # 2. Extract Class Centers (Weights from ArcFace Head)
    # The weights in ArcMarginProduct are (out_features, in_features)
    class_centers = model.arcface.weight.data
    class_centers = F.normalize(class_centers, p=2, dim=1).to(device)

    # 3. Compute Similarity Matrix
    # Shape: (N_Val, Num_Classes)
    similarity = torch.matmul(embeddings, class_centers.t())

    # 4. Get Top-5 Predictions
    _, top_k_indices = torch.topk(similarity, k=5, dim=1)

    # 5. Calculate AP per sample
    # Expand targets to match top_k shape: (N, 1) -> (N, 5)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_indices)

    # Check hits (boolean mask)
    hits = top_k_indices == targets_expanded

    # Ranks weights: 1, 1/2, 1/3, 1/4, 1/5
    ranks = torch.tensor([1.0, 0.5, 1 / 3, 0.25, 0.2], device=device)

    # AP for each sample is the sum of hits weighted by rank.
    # Since there is only 1 correct label per sample, this sums to at most one rank value.
    ap_scores = torch.sum(hits.float() * ranks, dim=1)

    map5 = ap_scores.mean().item()

    return map5, ap_scores.cpu().numpy(), targets.cpu().numpy()


def perform_failure_analysis(ap_scores, targets, idx_to_hotel):
    """
    Correlates Error Magnitude (1 - AP) with Class Frequency.
    """
    print("\n--- Failure Analysis ---")

    # Load training metadata to get class counts
    train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts = train_df["hotel_id"].value_counts().to_dict()

    # Map target indices back to hotel_ids, then to their frequency in training set
    # idx_to_hotel is a numpy array where index -> hotel_id
    target_hotel_ids = idx_to_hotel[targets]
    target_counts = [class_counts.get(hid, 0) for hid in target_hotel_ids]

    # Error Magnitude = 1.0 - Average Precision
    # High error (1.0) means the correct class was not in Top-5.
    # Low error (0.0) means the correct class was Rank 1.
    errors = 1.0 - ap_scores

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"error": errors, "class_count": target_counts})

    # Calculate correlation
    correlation = analysis_df["error"].corr(analysis_df["class_count"])

    print(
        f"Correlation between Error Magnitude and Class Frequency: {correlation:.10f}"
    )

    return correlation


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline execution
    Config.NUM_EPOCHS = 5
    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={device}"
    )

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, idx_to_hotel = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = HotelRecognitionModel()
    model.to(device)

    # 4. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Validation & Scoring
    print("Performing final validation...")
    # Calculate MAP@5
    val_map5, ap_scores, targets = calculate_map5(model, val_loader, device)

    # Print Metric in required format
    print(f"Final Validation Metric: {val_map5}")

    # 7. Failure Analysis
    perform_failure_analysis(ap_scores, targets, idx_to_hotel)

    # 8. Submission Generation
    threshold = 0.14571255006929015
    if val_map5 > threshold:
        print(f"Validation metric {val_map5} > {threshold}. Generating submission...")
        # engine.generate_submission loads the model from Config.MODEL_SAVE_PATH
        generate_submission()
    else:
        print(f"Validation metric {val_map5} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    run()
