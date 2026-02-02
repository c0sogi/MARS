import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.neighbors import NearestNeighbors

# Import provided library modules
from library.config import Config
from library.utils import set_seed, compute_map5
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline and Path Requirements
    Config.epochs = 30  # Increased to allow convergence for ArcFace
    Config.submission_path = "./submission/submission.csv"
    Config.debug = False  # Ensure we use the full dataset for best performance

    # Ensure output directories exist
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set reproducibility
    set_seed(Config.seed)

    print(f"Configuration:")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Device: {Config.device}")
    print(f"  Submission Path: {Config.submission_path}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Loading Data...")
    train_loader, gallery_loader, val_loader, test_loader, encoder = get_dataloaders(
        debug=Config.debug, load_cached_data=True
    )
    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 3. Training
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Initializing Trainer & Starting Training...")
    trainer = Trainer(train_loader, gallery_loader, val_loader, encoder)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 4. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Performing Final Validation...")
    # Load the best checkpoint saved during training
    if os.path.exists(Config.model_save_path):
        print(f"  Loading best model from {Config.model_save_path}")
        trainer.model.load_state_dict(
            torch.load(Config.model_save_path, map_location=Config.device)
        )
    else:
        print("  Warning: No checkpoint found. Using model state from last epoch.")

    # Compute final metric
    final_metric = trainer.validate()
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Performing Failure Analysis...")

    # Extract embeddings for detailed analysis
    # Note: trainer.validate() does this internally but returns only the score.
    # We repeat extraction to get raw data for correlation analysis.
    gallery_emb, gallery_labels = trainer.extract_embeddings(gallery_loader)
    query_emb, query_labels = trainer.extract_embeddings(val_loader)

    if len(gallery_emb) > 0 and len(query_emb) > 0:
        # KNN Search
        knn = NearestNeighbors(n_neighbors=Config.knn_k, metric="cosine", n_jobs=-1)
        knn.fit(gallery_emb)
        _, indices = knn.kneighbors(query_emb)
        predicted_labels = gallery_labels[indices]

        # Calculate per-sample error magnitude
        # Error Magnitude = 1.0 - Score (where Score is 1/(rank+1) or 0)
        error_magnitudes = []
        for truth, preds in zip(query_labels, predicted_labels):
            top_preds = list(preds)[:5]
            if truth in top_preds:
                rank = top_preds.index(truth)
                score = 1.0 / (rank + 1)
            else:
                score = 0.0
            error_magnitudes.append(1.0 - score)

        error_magnitudes = np.array(error_magnitudes)

        # Collect Input Features
        # 1. Image Dimensions (Width, Height, Aspect Ratio)
        val_df = val_loader.dataset.df
        widths = []
        heights = []

        # Read image dimensions directly
        for fp in val_df["file_path"]:
            full_path = os.path.join(Config.input_dir, fp)
            if os.path.exists(full_path):
                # Read header only if possible, but cv2 reads full. Fast enough for 451 imgs.
                img = cv2.imread(full_path)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                else:
                    widths.append(0)
                    heights.append(0)
            else:
                widths.append(0)
                heights.append(0)

        widths = np.array(widths)
        heights = np.array(heights)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            aspect_ratios = np.divide(widths, heights)
            aspect_ratios[~np.isfinite(aspect_ratios)] = 0

        # 2. Class Frequency (in Training Set)
        train_df = pd.read_csv(Config.train_csv_path)
        class_counts = train_df["Id"].value_counts().to_dict()
        # Map integer labels back to string IDs
        # Cite debug_lesson_5: Sanitize Sentinel Labels Before Inverse Transformation
        val_ids = [
            encoder.classes_[x] if x != -1 else "new_whale" for x in query_labels
        ]
        class_freqs = np.array([class_counts.get(x, 0) for x in val_ids])

        # Compute Correlations
        # Filter valid images
        valid_mask = (widths > 0) & (heights > 0)

        if np.sum(valid_mask) > 1:
            # Helper for correlation
            def calc_corr(x, y):
                if np.std(x) == 0 or np.std(y) == 0:
                    return 0.0
                return np.corrcoef(x, y)[0, 1]

            corr_w = calc_corr(error_magnitudes[valid_mask], widths[valid_mask])
            corr_h = calc_corr(error_magnitudes[valid_mask], heights[valid_mask])
            corr_ar = calc_corr(error_magnitudes[valid_mask], aspect_ratios[valid_mask])
            corr_freq = calc_corr(error_magnitudes[valid_mask], class_freqs[valid_mask])

            print(f"  Correlation (Error vs Width): {corr_w:.4f}")
            print(f"  Correlation (Error vs Height): {corr_h:.4f}")
            print(f"  Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")
            print(f"  Correlation (Error vs Class Frequency): {corr_freq:.4f}")
        else:
            print("  Insufficient valid data for correlation analysis.")
    else:
        print("  Skipping failure analysis (empty embeddings).")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Checking Submission Criteria...")
    threshold = 0.8543859649122806

    if final_metric > threshold:
        print(
            f"  Metric ({final_metric}) > Threshold ({threshold}). Proceeding to submission."
        )

        # Clean up memory
        del trainer
        del gallery_emb
        del query_emb
        torch.cuda.empty_cache()

        # Initialize Inference Engine
        inference = InferenceEngine(checkpoint_path=Config.model_save_path)

        # Generate predictions with Query Expansion
        # Note: This method automatically saves to Config.submission_path
        inference.predict_with_qe(
            test_loader=test_loader,
            gallery_loader=gallery_loader,
            encoder=encoder,
            load_cached_data=True,
        )
    else:
        print(
            f"  Metric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
