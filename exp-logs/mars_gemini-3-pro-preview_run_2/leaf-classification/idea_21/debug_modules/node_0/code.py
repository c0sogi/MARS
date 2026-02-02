import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

# Set random seed for reproducibility
np.random.seed(42)

# Import library components
from library.config import RANDOM_SEED, MORPHOLOGICAL_COLS, METADATA_DIR, TRAIN_FILE
from library.feature_engineering import MorphologyExtractor
from library.data_loader import LeafDataManager
from library.models import ExpertFactory
from library.ensemble import GreedyEnsembleSelector


def run_demo():
    print("Starting Library Usage Demo...")
    print("=" * 40)

    # =========================================================================
    # 1. DEMONSTRATE MORPHOLOGY EXTRACTOR
    # =========================================================================
    print("\n[1] Testing MorphologyExtractor...")

    # Load train metadata to get a valid image path
    df_train = pd.read_csv(TRAIN_FILE)
    sample_row = df_train.iloc[0]
    sample_image_path = sample_row["image_path"]

    extractor = MorphologyExtractor()
    features = extractor.extract_single_image(sample_image_path)

    print(f"  - Extracted features for {sample_image_path}")

    # Validation
    assert isinstance(features, dict), "Features should be returned as a dictionary."
    assert all(
        col in features for col in MORPHOLOGICAL_COLS
    ), "Missing morphological columns in extracted features."
    assert features["aspect_ratio"] >= 0, "Aspect ratio must be non-negative."

    print("  - MorphologyExtractor validation passed.")

    # =========================================================================
    # 2. DEMONSTRATE DATA LOADER
    # =========================================================================
    print("\n[2] Testing LeafDataManager...")

    data_manager = LeafDataManager()

    # Get splits (Train, Val, Test)
    # This handles feature extraction caching and preprocessing (PowerTransformer)
    print("  - Loading and preprocessing data splits...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data_manager.get_splits(
        load_cached_data=True
    )

    # Validation
    expected_views = ["original", "morphological", "combined"]
    assert isinstance(X_train, dict), "X_train should be a dictionary of views."
    assert all(
        view in X_train for view in expected_views
    ), "Missing expected views in X_train."

    n_train = len(y_train)
    n_val = len(y_val)
    n_classes = len(classes)

    assert X_train["original"].shape[0] == n_train, "X_train sample count mismatch."
    assert X_val["original"].shape[0] == n_val, "X_val sample count mismatch."

    print(f"  - Data loaded successfully.")
    print(f"    Train samples: {n_train}, Val samples: {n_val}, Classes: {n_classes}")
    print("  - LeafDataManager validation passed.")

    # =========================================================================
    # 3. DEMONSTRATE MODEL FACTORY & TRAINING
    # =========================================================================
    print("\n[3] Testing ExpertFactory and Model Training...")

    # Create Expert A: LDA on Original Features
    lda_expert = ExpertFactory.create_lda_expert()
    print("  - Created LDA Expert.")

    # Create Expert B: Calibrated Logistic Regression on Morphological Features
    lr_expert = ExpertFactory.create_calibrated_lr_expert()
    print("  - Created Calibrated LR Expert.")

    # Train Expert A
    print("  - Training LDA on 'original' view...")
    lda_expert.fit(X_train["original"], y_train)

    # Train Expert B
    print("  - Training LR on 'morphological' view...")
    lr_expert.fit(X_train["morphological"], y_train)

    # Generate Predictions on Validation Set
    preds_lda = lda_expert.predict_proba(X_val["original"])
    preds_lr = lr_expert.predict_proba(X_val["morphological"])

    # Validation
    assert preds_lda.shape == (n_val, n_classes), "LDA prediction shape mismatch."
    assert preds_lr.shape == (n_val, n_classes), "LR prediction shape mismatch."

    # Check probability range
    assert np.all((preds_lda >= 0) & (preds_lda <= 1)), "LDA probs out of range [0, 1]."

    print("  - Model training and prediction validation passed.")

    # =========================================================================
    # 4. DEMONSTRATE ENSEMBLE SELECTION
    # =========================================================================
    print("\n[4] Testing GreedyEnsembleSelector...")

    # Prepare dictionary of expert predictions
    expert_preds_dict = {"Expert_LDA_Original": preds_lda, "Expert_LR_Morph": preds_lr}

    # Initialize Selector (Low iterations for demo speed)
    selector = GreedyEnsembleSelector(n_iterations=5)

    # Fit the ensemble weights based on validation performance
    print("  - Fitting ensemble selector...")
    selector.fit(expert_preds_dict, y_val, classes=classes)

    # Predict using the fitted ensemble
    ensemble_preds = selector.predict(expert_preds_dict)

    # Validation
    assert ensemble_preds.shape == (
        n_val,
        n_classes,
    ), "Ensemble prediction shape mismatch."
    assert selector.weights, "Ensemble weights should not be empty after fitting."

    # Verify that weights sum to the number of iterations (logic of the greedy selector provided)
    total_weight = sum(selector.weights.values())
    assert (
        total_weight == 5
    ), f"Total weight {total_weight} does not match n_iterations 5."

    print("  - Ensemble selection validation passed.")

    print("\n" + "=" * 40)
    print("Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
