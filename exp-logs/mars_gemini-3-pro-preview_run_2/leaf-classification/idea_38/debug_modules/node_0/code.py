import sys
import os
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append("./")

from library.config import Config
from library.utils import set_seed, clipped_log_loss
from library.features import extract_single_image_features, get_morphometric_features
from library.data_loader import load_datasets
from library.preprocessors import get_preprocessor
from library.models import get_expert_model
from library.ensemble import GreedySelector
from library.pipeline import train_and_predict_expert, run_mrgde_pipeline


def main():
    # 1. Setup and Configuration
    print("=== 1. Setup and Configuration ===")
    set_seed(Config.RANDOM_SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Input Directory: {Config.INPUT_DIR}")

    # Verify directories exist (created by Config import or pre-existing)
    assert os.path.exists(Config.WORKING_DIR), "Working directory should exist."
    assert os.path.exists(Config.INPUT_DIR), "Input directory must exist."
    assert os.path.exists(Config.METADATA_DIR), "Metadata directory must exist."

    # 2. Feature Extraction (Single Image)
    print("\n=== 2. Testing Feature Extraction ===")
    # Pick a sample image from metadata
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    df_train = pd.read_csv(train_meta_path)
    sample_rel_path = df_train.iloc[0][Config.IMAGE_PATH_COL]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    print(f"Extracting features from: {sample_rel_path}")
    features = extract_single_image_features(sample_full_path)

    print(f"Extracted feature shape: {features.shape}")
    print(f"Extracted feature values: {features}")

    # Verify shape (11 features: 7 Hu + 4 Scalars)
    assert features.shape == (11,), "Feature vector must have 11 elements."
    assert features.dtype == Config.FLOAT_PRECISION, "Features must be float64."
    # Basic sanity check: features shouldn't be all zeros for a valid image
    if os.path.exists(sample_full_path):
        assert np.any(
            features != 0
        ), "Features should not be all zero for a valid image."

    # 3. Data Loading
    print("\n=== 3. Testing Data Loading ===")
    # We force load_cached_data=False initially to test the computation logic
    # Note: This computes morphometric features for all images (~1 min max)
    data = load_datasets(load_cached_data=False)

    # Verify structure
    assert "train" in data
    assert "val" in data
    assert "test" in data
    assert "classes" in data

    # Verify Train Data
    X_train_global = data["train"]["views"][Config.VIEW_GLOBAL]
    X_train_combined = data["train"]["views"][Config.VIEW_COMBINED]
    y_train = data["train"]["y"]

    print(f"Train Global View Shape: {X_train_global.shape}")
    print(f"Train Combined View Shape: {X_train_combined.shape}")
    print(f"Train Labels Shape: {y_train.shape}")
    print(f"Number of Classes: {len(data['classes'])}")

    assert X_train_global.shape[0] == y_train.shape[0], "X and y row counts must match."
    assert (
        X_train_combined.shape[1] == X_train_global.shape[1] + 11
    ), "Combined view should have 11 more features."
    assert len(data["classes"]) == 99, "Dataset should have 99 classes."

    # 4. Preprocessing
    print("\n=== 4. Testing Preprocessors ===")
    # Test Robust Quantile Transformer
    preprocessor = get_preprocessor(Config.BASIS_ROBUST)

    # Fit on a small subset
    subset_X = X_train_global[:100]
    preprocessor.fit(subset_X)
    transformed_X = preprocessor.transform(subset_X)

    print(f"Original Mean (feat 0): {subset_X[:, 0].mean():.4f}")
    print(f"Transformed Mean (feat 0): {transformed_X[:, 0].mean():.4f}")

    # Output should be roughly Gaussian (mean ~ 0, std ~ 1)
    assert transformed_X.shape == subset_X.shape
    assert (
        abs(transformed_X.mean()) < 0.5
    ), "Transformed data should be roughly centered."

    # 5. Model Instantiation & Training (Single Expert)
    print("\n=== 5. Testing Expert Model (LDA Fixed) ===")
    model = get_expert_model(Config.MODEL_LDA_FIXED)

    # Define a simple expert config
    expert_config = {
        "model": Config.MODEL_LDA_FIXED,
        "basis": Config.BASIS_PARAMETRIC,
        "view": Config.VIEW_GLOBAL,
        "id": "test_expert",
    }

    # Use train_and_predict_expert wrapper
    # We use a subset for speed in this unit test
    X_tr_sub = data["train"]["views"][Config.VIEW_GLOBAL]
    y_tr_sub = data["train"]["y"]
    X_val_sub = data["val"]["views"][Config.VIEW_GLOBAL]
    y_val_sub = data["val"]["y"]

    print("Training single expert on full train set...")
    preds = train_and_predict_expert(expert_config, X_tr_sub, y_tr_sub, X_val_sub)

    print(f"Prediction Shape: {preds.shape}")

    # Verify probability properties
    row_sums = preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities must sum to 1."
    assert preds.min() >= 0 and preds.max() <= 1.0, "Probabilities must be in [0, 1]."

    score = clipped_log_loss(y_val_sub, preds)
    print(f"Single Expert Log Loss: {score:.4f}")

    # 6. Ensemble Selection Logic
    print("\n=== 6. Testing Greedy Selector ===")
    # Create synthetic predictions to test selector logic
    # Expert A: Good predictions (add noise to truth)
    # Expert B: Random predictions

    n_val = len(y_val_sub)
    n_classes = len(data["classes"])

    # Create one-hot ground truth
    y_onehot = np.zeros((n_val, n_classes))
    y_onehot[np.arange(n_val), y_val_sub] = 1.0

    # Expert A: Ground truth mixed with small noise (Good)
    expert_a_preds = y_onehot * 0.8 + np.random.rand(n_val, n_classes) * 0.2
    expert_a_preds /= expert_a_preds.sum(axis=1, keepdims=True)

    # Expert B: Uniform random (Bad)
    expert_b_preds = np.random.rand(n_val, n_classes)
    expert_b_preds /= expert_b_preds.sum(axis=1, keepdims=True)

    preds_dict = {"expert_A": expert_a_preds, "expert_B": expert_b_preds}

    selector = GreedySelector(max_iterations=10, tolerance=1e-4)
    weights = selector.fit(preds_dict, y_val_sub)

    print(f"Selected Weights: {weights}")
    print(f"Best Score: {selector.best_score:.4f}")

    # Expert A should be selected and have higher weight than B (or B might not be selected at all)
    assert "expert_A" in weights, "The better expert should be selected."
    assert weights.get("expert_A", 0) >= weights.get(
        "expert_B", 0
    ), "Better expert should have >= weight."

    # Test Prediction
    ensemble_preds = selector.predict(preds_dict)
    assert ensemble_preds.shape == expert_a_preds.shape

    # 7. Full Pipeline Execution
    print("\n=== 7. Running Full MRGDE Pipeline ===")
    # This runs the actual pipeline defined in library/pipeline.py
    # It will:
    # 1. Load data (using cache if available)
    # 2. Train all experts defined in Config.get_expert_library()
    # 3. Select best ensemble
    # 4. Retrain on Train+Val
    # 5. Predict on Test
    # 6. Save submission.csv

    # To ensure a clean run for the demo, we allow it to use the cache generated in step 3
    run_mrgde_pipeline(load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")

    # Verify submission format
    assert "id" in df_sub.columns, "Submission must have 'id' column."
    assert len(df_sub) == 99, "Submission must have 99 rows (test set size)."
    assert df_sub.shape[1] == 100, "Submission must have 100 columns (id + 99 species)."

    # Verify values
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values
    assert probs.min() >= 0, "Probabilities cannot be negative."
    assert probs.max() <= 1, "Probabilities cannot be > 1."

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
