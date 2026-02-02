import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Ensure current directory is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, apk
from library.dataset import get_dataloaders
from library.model import HotelConvNeXt
from library.trainer import train_one_epoch, extract_features, validate
import library.post_processing as pp

# =========================================================================
# Configuration Overrides for Fast Baseline
# =========================================================================
Config.epochs = 3  # Limit epochs to ensure execution within time limits
Config.submission_path = "./submission/submission.csv"
# Ensure submission directory exists
os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)


def run():
    # 1. Initialization
    seed_everything(Config.seed)
    print(f"Running on device: {Config.device}")

    # 2. Data Loading
    # We use the full dataset (debug=False) to ensure we can hit the target metric,
    # but we limit the number of epochs to keep it fast.
    print("Loading data...")
    train_loader, val_loader, test_loader, gallery_loader, num_classes = (
        get_dataloaders(debug=False)
    )
    print(f"Data loaded. Number of classes: {num_classes}")

    # 3. Model Setup
    model = HotelConvNeXt(
        num_classes=num_classes,
        k_subcenters=Config.k_subcenters,
        margin=Config.margin,
        scale=Config.scale,
    ).to(Config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    # 4. Training Loop
    best_map = 0.0
    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )
        scheduler.step()

        # Validate
        val_map = validate(model, val_loader, gallery_loader, Config.device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.epochs} | Loss: {train_loss:.4f} | Val MAP@5: {val_map:.5f} | Time: {elapsed:.0f}s"
        )

        # Checkpoint
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.best_model_path)
            print(f"  New best model saved! (MAP@5: {best_map:.5f})")

    print(f"Training complete. Best MAP@5: {best_map:.5f}")

    # 5. Final Validation & Failure Analysis
    print("\n=== Final Validation & Failure Analysis ===")

    # Load best model
    model.load_state_dict(torch.load(Config.best_model_path))
    model.eval()

    # Extract embeddings
    print("Extracting embeddings for analysis...")
    gal_emb, gal_labels, gal_names = extract_features(
        model, gallery_loader, Config.device
    )
    val_emb, val_labels, val_names = extract_features(model, val_loader, Config.device)

    # Normalize
    gal_emb = F.normalize(gal_emb, dim=1).to(Config.device)
    val_emb = F.normalize(val_emb, dim=1).to(Config.device)

    # Compute Similarity Matrix (Val x Gallery)
    # Note: On A100 40GB, this matmul fits in memory.
    sim_matrix = torch.matmul(val_emb, gal_emb.T)

    # Get Top 5 predictions
    _, indices = torch.topk(sim_matrix, k=5, dim=1)
    indices = indices.cpu().numpy()

    gal_labels_np = gal_labels.numpy()
    val_labels_np = val_labels.numpy()

    # Calculate per-sample AP@5
    ap_scores = []
    for i in range(len(indices)):
        preds = gal_labels_np[indices[i]]
        actual = [val_labels_np[i]]
        score = apk(actual, list(preds), k=5)
        ap_scores.append(score)

    final_metric = np.mean(ap_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation with Train Samples
    print("Calculating correlation with training class frequency...")
    val_metadata = pd.read_csv(Config.val_metadata_path)
    train_metadata = pd.read_csv(Config.train_metadata_path)

    # Create a DataFrame for analysis
    # Ensure alignment: val_names comes from loader, which is sequential based on dataset
    analysis_df = pd.DataFrame({"image": val_names, "ap_score": ap_scores})

    # Merge with metadata to get hotel_id
    analysis_df = analysis_df.merge(
        val_metadata[["image", "hotel_id"]], on="image", how="left"
    )

    # Calculate class counts from training set
    class_counts = train_metadata["hotel_id"].value_counts()

    # Map counts to validation samples
    analysis_df["train_samples"] = analysis_df["hotel_id"].map(class_counts).fillna(0)

    # Calculate correlation
    # We correlate Error Magnitude (1 - Score) with Features.
    # Or simply Score with Features.
    # High score = Low error.
    corr_score_freq = analysis_df["ap_score"].corr(analysis_df["train_samples"])

    print(
        f"Correlation between Model Performance (AP) and Training Samples: {corr_score_freq:.4f}"
    )
    print(
        f"Correlation between Error Magnitude and Training Samples: {-corr_score_freq:.4f}"
    )

    # 6. Submission
    threshold = 0.7120973100214514
    if final_metric > threshold:
        print(
            f"\nMetric {final_metric} > {threshold}. Proceeding to submission generation..."
        )

        # Extract Test Embeddings
        print("Extracting Test embeddings...")
        test_emb, _, test_names = extract_features(model, test_loader, Config.device)

        # Prepare DataFrames for Post-Processing module
        # Convert tensors to list of numpy arrays for storage
        gal_emb_np = gal_emb.cpu().numpy()
        test_emb_np = test_emb.cpu().numpy()

        gal_df = pd.DataFrame({"image": gal_names})
        gal_df["embedding"] = list(gal_emb_np)

        test_df = pd.DataFrame({"image": test_names})
        test_df["embedding"] = list(test_emb_np)

        # Save to Parquet (required by pp module)
        print("Saving embeddings to Parquet...")
        gal_df.to_parquet(Config.gallery_embeddings_path, index=False)
        test_df.to_parquet(Config.query_embeddings_path, index=False)

        # Run Post-Processing
        print("Running post-processing (DBA + QE)...")
        # load_cached_data=False forces it to use the new parquet files we just wrote
        pp.run_post_processing(load_cached_data=False)

    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
