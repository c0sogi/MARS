import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, apk, mapk
from library.dataset import get_dataloaders
from library.model import EfficientNetArcFace
from library.trainer import train_fn
from library.inference import inference_fn


def validate_and_analyze(model, val_loader, device, train_df):
    """
    Performs validation, computes MAP@5, and runs failure analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Get normalized class centers from the ArcFace head
    centers = F.normalize(model.head.weight, p=2, dim=1)

    # Inference on Validation Set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Get embeddings
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Compute Cosine Similarity
            logits = torch.matmul(embeddings, centers.T)

            # Get top 5 predictions
            _, top_indices = logits.topk(Config.TOP_K, dim=1)

            all_preds.extend(top_indices.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute MAP@5
    score = mapk(all_labels, all_preds, k=Config.TOP_K)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {score}")

    # --- Failure Analysis ---
    # Calculate AP@5 per sample to define error magnitude
    # Error magnitude = 1.0 - AP@5
    ap_scores = [apk(a, p, k=Config.TOP_K) for a, p in zip(all_labels, all_preds)]
    error_magnitude = [1.0 - ap for ap in ap_scores]

    # Create a DataFrame for analysis
    # We retrieve metadata from the dataset
    val_df = val_loader.dataset.df.copy()
    val_df["error"] = error_magnitude

    # Feature 1: Chain ID
    # Check correlation between error and chain ID
    if "chain" in val_df.columns:
        # Fill NaNs if any (though metadata analysis showed none)
        chain_vals = val_df["chain"].fillna(0).astype(float)
        corr_chain = val_df["error"].corr(chain_vals)
        print(f"Correlation between Error and Chain ID: {corr_chain}")

    # Feature 2: Class Frequency in Training Set
    # We need to know how many times each class appeared in training
    freq_map = train_df["label_idx"].value_counts().to_dict()
    val_df["train_freq"] = val_df["label_idx"].map(freq_map).fillna(0)

    corr_freq = val_df["error"].corr(val_df["train_freq"])
    print(f"Correlation between Error and Training Class Frequency: {corr_freq}")

    return score


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline
    Config.EPOCHS = 6  # Sufficient for convergence on A100 without exceeding time
    Config.DEBUG = False  # Use full dataset

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading Data...")
    # Load cached data if available (library handles this)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    # Get train dataframe for frequency analysis later
    train_df = train_loader.dataset.df

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print(f"Initializing Model: {Config.MODEL_NAME} with {num_classes} classes...")
    model = EfficientNetArcFace(n_classes=num_classes).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    best_map = 0.0
    print(f"Starting Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train Step
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch)

        # Validation Step (Inline to track best model)
        model.eval()
        centers = F.normalize(model.head.weight, p=2, dim=1)
        ep_preds = []
        ep_labels = []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                embs = model(imgs)
                embs = F.normalize(embs, p=2, dim=1)
                logits = torch.matmul(embs, centers.T)
                _, top_k = logits.topk(Config.TOP_K, dim=1)
                ep_preds.extend(top_k.cpu().numpy())
                ep_labels.extend(lbls.numpy())

        val_map = mapk(ep_labels, ep_preds, k=Config.TOP_K)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss:.4f} | Val MAP@5: {val_map:.4f} | Time: {elapsed:.0f}s"
        )

        # Save Best Model
        if val_map > best_map:
            best_map = val_map
            print(f"New Best Score! Saving model to {Config.MODEL_PATH}")
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ---------------------------------------------------------
    # 5. Final Evaluation & Analysis
    # ---------------------------------------------------------
    print("\nTraining Complete. Loading best model for analysis...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model saved (did not improve?). Using last state.")

    final_score = validate_and_analyze(model, val_loader, device, train_df)

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.5589516758918762

    if final_score > THRESHOLD:
        print(
            f"\nValidation Score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Label Encoder for mapping indices back to Hotel IDs
        encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.parquet")
        if os.path.exists(encoder_path):
            label_df = pd.read_parquet(encoder_path)
            label_map = dict(zip(label_df["hotel_id"], label_df["label_idx"]))

            # Run Inference
            inference_fn(test_loader, model, device, label_map)
        else:
            print("Error: Label encoder not found. Cannot generate submission.")
    else:
        print(
            f"\nValidation Score ({final_score}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
