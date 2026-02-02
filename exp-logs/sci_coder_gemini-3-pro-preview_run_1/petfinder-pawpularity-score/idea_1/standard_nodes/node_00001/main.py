import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.dataset import PawpularityDataset, get_transforms
from library.feature_extractor import MobileNetExtractor
from library.regressor import RidgeHead


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Starting Fast Baseline: Linear Probing with MobileNetV3...")

    # 2. Data Preparation
    # Define transforms
    transforms = get_transforms(Config.IMG_SIZE)

    # Initialize Datasets
    # We use the full dataset as MobileNet inference is very fast (~1 minute for 7k images on A100)
    train_dataset = PawpularityDataset(
        metadata_path=Config.TRAIN_METADATA,
        image_root=Config.INPUT_ROOT,
        transform=transforms,
        test_mode=False,
    )

    val_dataset = PawpularityDataset(
        metadata_path=Config.VAL_METADATA,
        image_root=Config.INPUT_ROOT,
        transform=transforms,
        test_mode=False,
    )

    test_dataset = PawpularityDataset(
        metadata_path=Config.TEST_METADATA,
        image_root=Config.INPUT_ROOT,
        transform=transforms,
        test_mode=True,
    )

    # Initialize DataLoaders
    # num_workers=0 to avoid potential multiprocessing overhead/issues in this specific env,
    # though Config suggests 4. Using 2 for safety and speed balance.
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Feature Extraction
    extractor = MobileNetExtractor()

    print("Extracting/Loading Train Features...")
    train_feats, train_meta, train_targets = extractor.extract_features(
        dataloader=train_loader,
        cache_features_path=Config.CACHE_TRAIN_FEATURES,
        cache_aux_path=Config.CACHE_TRAIN_TARGETS,
        load_cached_data=True,
        is_test=False,
    )

    print("Extracting/Loading Validation Features...")
    val_feats, val_meta, val_targets = extractor.extract_features(
        dataloader=val_loader,
        cache_features_path=Config.CACHE_VAL_FEATURES,
        cache_aux_path=Config.CACHE_VAL_TARGETS,
        load_cached_data=True,
        is_test=False,
    )

    print("Extracting/Loading Test Features...")
    test_feats, test_meta, test_ids = extractor.extract_features(
        dataloader=test_loader,
        cache_features_path=Config.CACHE_TEST_FEATURES,
        cache_aux_path=Config.CACHE_TEST_IDS,
        load_cached_data=True,
        is_test=True,
    )

    # 4. Model Training
    print("Training Ridge Regression Head...")
    model = RidgeHead(alpha=Config.RIDGE_ALPHA)
    model.fit(train_feats, train_meta, train_targets)

    # 5. Validation & Evaluation
    print("Evaluating on Validation Set...")
    # Calculate RMSE
    val_rmse = model.evaluate(val_feats, val_meta, val_targets)
    print(f"Final Validation Metric: {val_rmse}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_preds = model.predict(val_feats, val_meta)

    # Calculate absolute errors
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for correlation analysis
    # We use the metadata columns from the dataset class
    meta_cols = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    analysis_df = pd.DataFrame(val_meta, columns=meta_cols)
    analysis_df["Error_Magnitude"] = errors

    # Compute correlation
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")
    print("Correlation between Error Magnitude and Metadata Features:")
    print(correlations.sort_values(ascending=False).to_string())

    # 7. Submission
    print("\nGenerating Submission...")
    test_preds = model.predict(test_feats, test_meta)

    # Ensure predictions are within valid range [1, 100] (optional but good practice)
    test_preds = np.clip(test_preds, 1.0, 100.0)

    model.save_submission(test_ids, test_preds, Config.SUBMISSION_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
