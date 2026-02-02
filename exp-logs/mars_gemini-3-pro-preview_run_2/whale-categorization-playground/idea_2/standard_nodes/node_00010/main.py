import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Import provided library modules
import library.config as config
from library.trainer import Trainer
from library.inference import InferenceManager
from library.config import seed_everything


def main():
    # -------------------------------------------------------------------------
    # 0. Setup & Configuration
    # -------------------------------------------------------------------------
    # Override configuration
    # We use the config defaults (25 epochs) as optimized in config.py
    # config.NUM_EPOCHS = 25

    # Ensure reproducibility
    seed_everything(config.SEED)

    # -------------------------------------------------------------------------
    # 1. Training
    # -------------------------------------------------------------------------
    # Initialize Trainer with caching enabled for speed
    trainer = Trainer(load_cached_data=True)

    # Execute training
    # This will train for config.NUM_EPOCHS and save 'best_model.pth'
    trainer.fit()

    # Retrieve the best validation metric achieved
    final_metric = trainer.best_map5

    # Print the required metric output
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 2. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Load the best model for analysis
    # We need to perform retrieval on the validation set manually to get per-sample errors
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Skipping failure analysis.")
    else:
        # Load model weights
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=config.DEVICE)
        )
        trainer.model.eval()

        # Extract features for Gallery (Train) and Query (Val)
        # Note: Trainer.train_loader excludes 'new_whale', which is correct for validation
        # against known classes as per the training setup.
        train_feats, train_labels = trainer.extract_features(trainer.train_loader)
        val_feats, val_labels = trainer.extract_features(trainer.val_loader)

        train_feats = train_feats.to(config.DEVICE)
        val_feats = val_feats.to(config.DEVICE)

        # Compute Similarity Matrix
        # (N_val, D) @ (D, N_train) -> (N_val, N_train)
        sim_matrix = torch.mm(val_feats, train_feats.t())

        # Retrieve Top 5
        _, topk_indices = torch.topk(sim_matrix, k=5, dim=1)

        topk_indices = topk_indices.cpu().numpy()
        train_labels_np = train_labels.numpy()
        val_labels_np = val_labels.numpy()

        # Calculate Error Magnitude
        # Error = 1.0 - (1 / Rank) if found, else 1.0
        errors = []
        for i, target in enumerate(val_labels_np):
            preds = train_labels_np[topk_indices[i]]
            score = 0.0
            if target in preds:
                # np.where returns tuple, get index of first match
                rank = np.where(preds == target)[0][0]
                score = 1.0 / (rank + 1)
            errors.append(1.0 - score)

        # Collect Input Features from Metadata/Files
        val_df = trainer.val_loader.dataset.df

        meta_features = {
            "file_size": [],
            "width": [],
            "height": [],
            "aspect_ratio": [],
            "error": errors,
        }

        for _, row in val_df.iterrows():
            fpath = os.path.join(config.INPUT_DIR, row["file_path"])

            # File Size
            try:
                fsize = os.path.getsize(fpath)
            except:
                fsize = 0

            # Dimensions
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
            except:
                w, h = 0, 0

            ar = w / h if h > 0 else 0

            meta_features["file_size"].append(fsize)
            meta_features["width"].append(w)
            meta_features["height"].append(h)
            meta_features["aspect_ratio"].append(ar)

        df_analysis = pd.DataFrame(meta_features)

        # Calculate and Print Correlations
        print("Correlation between Error Magnitude and Input Features:")
        for feat in ["file_size", "width", "height", "aspect_ratio"]:
            # Simple correlation ignoring NaNs/Zeros if any
            series = df_analysis[feat]
            if series.std() > 0:
                corr = series.corr(df_analysis["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: NaN (No variance)")

    # -------------------------------------------------------------------------
    # 3. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.59527

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Initialize Inference Manager
        manager = InferenceManager(checkpoint_name="best_model.pth")

        # Force re-computation of embeddings to ensure they match the trained model.
        # We delete the embedding cache files but keep the image cache files.
        for cache_name in ["gallery_embeddings.npy", "query_embeddings.npy"]:
            p = os.path.join(config.WORKING_DIR, cache_name)
            if os.path.exists(p):
                os.remove(p)

        # Run prediction
        # load_cached_data=True will load image caches (fast) but recompute embeddings (since we deleted them)
        manager.predict(load_cached_data=True, threshold=0.35)

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
