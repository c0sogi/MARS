import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2

# Import provided library components
from library.config import Config
from library.utils import seed_everything, save_checkpoint
from library.dataset import get_loaders
from library.model import WhaleModel
from library.train import train_one_epoch, extract_features, calculate_map5
from library.evaluate import inference


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Ensure submission directory exists
    if not os.path.exists("./submission"):
        os.makedirs("./submission")

    # Override Config
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    # Config.EPOCHS is now set to 30 in Config class (Cite solution_lesson_node_00019)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print(f"Initializing WhaleModel with {num_classes} classes...")
    model = WhaleModel(num_classes=num_classes).to(device)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_map5 = -1.0
    best_model_path = os.path.join(os.path.dirname(Config.MODEL_PATH), "model_best.pth")

    print(f"Starting Training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation (Retrieval-based)
        # Extract features for Gallery (Train) and Query (Val)
        gallery_feats, gallery_labels = extract_features(model, train_loader, device)
        query_feats, query_labels = extract_features(model, val_loader, device)

        # Calculate MAP@5
        val_map5 = calculate_map5(
            query_feats, query_labels, gallery_feats, gallery_labels, device
        )

        # Scheduler step
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MAP@5: {val_map5:.6f}"
        )

        # Save Best Model
        if val_map5 > best_map5:
            best_map5 = val_map5
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "best_map5": best_map5,
                },
                is_best=True,
                filepath=Config.MODEL_PATH,
            )

    print(f"Training finished. Best MAP@5: {best_map5:.6f}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Load best model
    if os.path.exists(best_model_path):
        print("Loading best model for analysis...")
        checkpoint = torch.load(
            best_model_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Best model not found, using current model.")

    model.eval()

    # Extract features again with best model for analysis
    gallery_feats, gallery_labels = extract_features(model, train_loader, device)
    query_feats, query_labels = extract_features(model, val_loader, device)

    # Move to GPU for calculation
    q_feats = query_feats.to(device)
    g_feats = gallery_feats.to(device)

    # Similarity Matrix
    sim_matrix = torch.mm(q_feats, g_feats.t())

    # Top 5 indices
    _, topk_indices = torch.topk(sim_matrix, k=5, dim=1)
    topk_indices = topk_indices.cpu().numpy()

    q_lbls = query_labels.numpy()
    g_lbls = gallery_labels.numpy()

    # Calculate Error Magnitude (1 - AP) per sample
    error_magnitudes = []
    widths = []
    heights = []
    aspect_ratios = []

    # Validation DataFrame for metadata
    val_df = val_loader.dataset.df
    root_dir = val_loader.dataset.root_dir

    for i in range(len(q_lbls)):
        true_label = q_lbls[i]
        pred_inds = topk_indices[i]
        pred_lbls = g_lbls[pred_inds]

        # Calculate AP
        ap = 0.0
        if true_label in pred_lbls:
            rank = np.where(pred_lbls == true_label)[0][0]
            ap = 1.0 / (rank + 1)

        error = 1.0 - ap
        error_magnitudes.append(error)

        # Get Image Metadata
        row = val_df.iloc[i]
        fpath = os.path.join(root_dir, row["file_path"])

        # Read image to get dimensions
        img = cv2.imread(fpath)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Compute Correlations
    if len(error_magnitudes) > 1:
        corr_w = np.corrcoef(error_magnitudes, widths)[0, 1]
        corr_h = np.corrcoef(error_magnitudes, heights)[0, 1]
        corr_ar = np.corrcoef(error_magnitudes, aspect_ratios)[0, 1]

        print(f"Correlation (Error vs Width): {corr_w:.6f}")
        print(f"Correlation (Error vs Height): {corr_h:.6f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.6f}")
    else:
        print("Insufficient data for correlation analysis.")

    # -------------------------------------------------------------------------
    # 6. Final Output & Submission
    # -------------------------------------------------------------------------
    print(f"Final Validation Metric: {best_map5}")

    SUBMISSION_THRESHOLD = 0.8543859649122806

    if best_map5 > SUBMISSION_THRESHOLD:
        print(
            f"Metric exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # Run inference (load_cached_data=False to use the loaded best model)
        inference(model, train_loader, test_loader, device, load_cached_data=False)
    else:
        print(
            f"Metric does not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
