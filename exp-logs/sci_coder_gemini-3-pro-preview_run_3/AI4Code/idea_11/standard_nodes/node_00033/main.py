import pandas as pd
import numpy as np
import torch
import gc
import os
import sys

# Import from provided library
from library.config import Config, set_seed
from library.backbone import BackboneTrainer
from library.feature_extractor import FeatureEngineer
from library.ranker import LGBMRanker
from library.inference import InferencePipeline


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    print("=== Starting Runfile Execution ===")

    # 2. Fine-Tune Backbone
    # This step aligns the vector space of code and markdown using Contrastive Learning.
    # Config.FT_SAMPLE_SIZE (40k) limits the data here for a fast baseline.
    print("\n[Step 1] Fine-Tuning Backbone...")
    backbone = BackboneTrainer()
    backbone.train(load_cached_data=True)

    # Clean up GPU memory before the heavy feature extraction phase
    del backbone
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Feature Extraction
    # We re-initialize FeatureEngineer, which will load the fine-tuned weights from disk.
    print("\n[Step 2] Feature Extraction...")
    fe = FeatureEngineer()

    # Extract/Load Train Features (Full Dataset)
    train_df = fe.extract_features(
        metadata_path=Config.TRAIN_PATH, mode="train", load_cached_data=True
    )

    # Extract/Load Validation Features
    val_df = fe.extract_features(
        metadata_path=Config.VAL_PATH, mode="val", load_cached_data=True
    )

    # Clean up feature engineer and backbone to free memory for LightGBM
    del fe
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Train Regressor
    print("\n[Step 3] Training LightGBM Regressor...")
    ranker = LGBMRanker()
    ranker.train(train_df, val_df)

    # 5. Validation Inference & Metric
    print("\n[Step 4] Validation Inference...")
    pipeline = InferencePipeline()
    # This calculates the official Kendall Tau on the validation set using the full pipeline logic
    val_score = pipeline.run_validation_inference(load_cached_features=True)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    print("\n[Step 5] Failure Analysis...")
    # Generate predictions on validation set to analyze systematic errors
    if not val_df.empty:
        # Get raw predictions from the ranker
        val_preds = ranker.predict(val_df)
        val_df["pred"] = val_preds

        # Calculate Error Magnitude
        val_df["error"] = np.abs(val_df["target"] - val_df["pred"])

        # Select numerical features for correlation analysis
        numeric_cols = val_df.select_dtypes(include=[np.number]).columns

        # Compute correlation of all features with the 'error' column
        corr_matrix = val_df[numeric_cols].corr()

        # Sort by absolute correlation to find strongest associations (positive or negative)
        error_corrs = (
            corr_matrix["error"].drop("error").sort_values(ascending=False, key=abs)
        )

        print("Correlation between Error Magnitude and Input Features (Top 10):")
        print(error_corrs.head(10))
    else:
        print("Validation dataframe is empty. Skipping failure analysis.")

    # 7. Submission
    print("\n[Step 6] Submission Check...")
    THRESHOLD = 0.8061

    if val_score > THRESHOLD:
        print(f"Validation metric {val_score} > {THRESHOLD}. Generating submission...")
        pipeline.run_test_inference(load_cached_features=True)
    else:
        print(f"Validation metric {val_score} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
