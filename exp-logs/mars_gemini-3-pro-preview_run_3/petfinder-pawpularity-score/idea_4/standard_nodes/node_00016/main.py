import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_score, save_submission
from library.data import load_dataset_dfs, get_dataloaders
from library.engine import extract_features, run_fine_tuning
from library.models import AdaptiveBackbone
from library.ensemble import StackingRegressor


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)

    print("==========================================")
    print("      Pet Pawpularity Inference Pipeline   ")
    print("==========================================")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[1/5] Loading Datasets...")
    train_df, val_df, test_df = load_dataset_dfs()

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)
    print(
        f"Data loaded: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})"
    )

    # ==========================================
    # 3. Stage 1: Fine-Tuning Backbones
    # ==========================================
    print("\n[2/5] Stage 1: Fine-Tuning Adaptive Backbones...")
    # This phase adapts the ImageNet weights to the specific visual features of the dataset
    model = run_fine_tuning(train_loader, val_loader, epochs=Config.EPOCHS)

    # ==========================================
    # 4. Stage 2: Feature Extraction
    # ==========================================
    print("\n[3/5] Stage 2: Feature Extraction (with TTA)...")
    # Extract embeddings using the fine-tuned model.
    # TTA (Test Time Augmentation) is enabled to robustify the features.

    # Extract Train Features
    print("Processing Training Set...")
    X_train, y_train, train_ids = extract_features(
        model, train_loader, mode="train", tta=True, load_cached_data=True
    )

    # Extract Validation Features
    print("Processing Validation Set...")
    X_val, y_val, val_ids = extract_features(
        model, val_loader, mode="valid", tta=True, load_cached_data=True
    )

    # Extract Test Features
    print("Processing Test Set...")
    X_test, _, test_ids = extract_features(
        model, test_loader, mode="test", tta=True, load_cached_data=True
    )

    # ==========================================
    # 5. Stage 2: Stacking Ensemble Training
    # ==========================================
    print("\n[4/5] Stage 2: Training Stacking Ensemble...")
    stacker = StackingRegressor(seed=Config.SEED)

    # Perform Cross-Validation to train Meta-Learner and fit Base Models
    stacker.cross_validate_and_fit(X_train, y_train)

    # ==========================================
    # 6. Evaluation & Failure Analysis
    # ==========================================
    print("\n[5/5] Evaluation & Analysis...")

    # Predict on Hold-out Validation Set
    val_preds = stacker.predict(X_val)
    val_rmse = get_score(y_val, val_preds)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {val_rmse}")

    # Failure Analysis: Correlation between Error and Metadata
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val - val_preds)

    # The last 12 columns of the feature matrix correspond to the metadata
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
    meta_features = X_val[:, -12:]

    print("Correlation between Absolute Error and Metadata Features:")
    for i, col in enumerate(meta_cols):
        feat_vals = meta_features[:, i]
        # Calculate correlation (handle constant features safely)
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        print(f"  {col}: {corr:.4f}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 18.09007350517167

    if val_rmse < THRESHOLD:
        print(f"\nValidation score ({val_rmse:.5f}) meets threshold ({THRESHOLD:.5f}).")
        print("Generating submission file...")

        test_preds = stacker.predict(X_test)
        save_submission(test_ids, test_preds)
    else:
        print(
            f"\nValidation score ({val_rmse:.5f}) does NOT meet threshold ({THRESHOLD:.5f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
