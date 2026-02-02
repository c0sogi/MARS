import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import from the provided library
from library.config import (
    RANDOM_SEED,
    TOPOLOGIES,
    SHRINKAGE_GRID,
    WORKING_DIR,
    INPUT_DIR,
    METADATA_DIR,
)
from library.utils import set_seed, calculate_log_loss
from library.image_features import ImageFeatureExtractor
from library.data_factory import load_data
from library.pipeline_factory import PipelineFactory
from library.ensemble_selector import GreedyEnsemble

# Ensure reproducibility
set_seed(RANDOM_SEED)


def demo_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n=== Demo: Utils ===")

    # Test calculate_log_loss with unnormalized probabilities
    # True label: class 0
    y_true = np.array([0, 1])
    # Preds: unnormalized, row 0 favors class 0, row 1 favors class 1
    y_pred = np.array([[10.0, 2.0, 1.0], [1.0, 5.0, 1.0]])

    # We expect the function to normalize these rows to sum to 1 before scoring
    # Row 0 -> [10/13, 2/13, 1/13] -> High prob for class 0
    # Row 1 -> [1/7, 5/7, 1/7] -> High prob for class 1
    # Since y_true matches the high probs, loss should be relatively low
    # Cite debug_lesson_7: Explicitly pass labels when validation data is sparse
    loss = calculate_log_loss(y_true, y_pred, labels=[0, 1, 2])

    print(f"Calculated Log Loss: {loss:.4f}")

    # Verification
    assert loss < 1.0, "Log loss should be low for correct predictions."
    assert isinstance(loss, float), "Log loss should return a float."
    print("Utils verification passed.")


def demo_image_features():
    """
    Demonstrates image feature extraction on a single image.
    """
    print("\n=== Demo: Image Features ===")

    extractor = ImageFeatureExtractor()

    # Pick an image from the input directory to test
    # We rely on the metadata to find a valid image
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df = pd.read_csv(train_meta_path)
        # Get first image path
        rel_path = df.iloc[0]["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            print(f"Extracting features from: {full_path}")
            features = extractor._extract_single_image_features(full_path)

            print(f"Extracted feature vector shape: {features.shape}")
            print(f"Feature values: {features}")

            # Verification
            assert features.shape == (11,), "Feature vector must have 11 elements."
            assert features.dtype == np.float64, "Features must be float64."
            # Check for non-zero values (assuming the image is not empty/corrupt)
            if np.sum(features) == 0:
                print(
                    "Warning: Features are all zero. Image might be empty or thresholding failed."
                )
            else:
                print("Features extracted successfully.")
        else:
            print(f"Image file not found at {full_path}. Skipping extraction test.")
    else:
        print("Metadata not found. Skipping extraction test.")


def demo_data_loading():
    """
    Demonstrates the data factory loading process.
    """
    print("\n=== Demo: Data Loading ===")

    # Load data (this handles caching internally)
    # We set load_cached_data=True to use existing cache if available, or create it.
    data = load_data(load_cached_data=True)

    # Verify structure
    required_keys = ["train", "val", "test", "classes"]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in loaded data."

    print(f"Classes found: {len(data['classes'])}")
    print(f"Train samples: {data['train']['X_global'].shape[0]}")
    print(f"Val samples:   {data['val']['X_global'].shape[0]}")
    print(f"Test samples:  {data['test']['X_global'].shape[0]}")

    # Verify alignment
    assert len(data["train"]["X_global"]) == len(
        data["train"]["y"]
    ), "Train X and y length mismatch."
    assert (
        data["train"]["X_global"].shape[1] == 192
    ), "Global features should have 192 columns."
    assert (
        data["train"]["X_morph"].shape[1] == 11
    ), "Morph features should have 11 columns."

    print("Data loading verification passed.")
    return data


def demo_pipelines_and_ensemble(data):
    """
    Demonstrates pipeline creation, training, and ensemble selection.
    """
    print("\n=== Demo: Pipelines & Ensemble ===")

    X_train_global = data["train"]["X_global"]
    X_train_morph = data["train"]["X_morph"]
    y_train = data["train"]["y"]

    X_val_global = data["val"]["X_global"]
    X_val_morph = data["val"]["X_morph"]
    y_val = data["val"]["y"]

    # For demonstration speed, we'll use a subset of the training data
    # and a fixed shrinkage parameter instead of the full grid.
    subset_size = 200
    if len(y_train) > subset_size:
        indices = np.random.choice(len(y_train), subset_size, replace=False)
        X_train_global_sub = X_train_global[indices]
        X_train_morph_sub = X_train_morph[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_global_sub = X_train_global
        X_train_morph_sub = X_train_morph
        y_train_sub = y_train

    print(f"Training on subset of {len(y_train_sub)} samples for speed...")

    library_preds_val = {}
    shrinkage = 0.1  # Fixed shrinkage for demo

    # 1. Train Topology A (Global Features)
    print("Training Topology A...")
    pipeline_a = PipelineFactory.create_pipeline("A", shrinkage)
    pipeline_a.fit(X_train_global_sub, y_train_sub)
    preds_a = pipeline_a.predict_proba(X_val_global)
    library_preds_val["Topo_A"] = preds_a

    # 2. Train Topology D (Morphometric Features)
    print("Training Topology D...")
    pipeline_d = PipelineFactory.create_pipeline("D", shrinkage)
    pipeline_d.fit(X_train_morph_sub, y_train_sub)
    preds_d = pipeline_d.predict_proba(X_val_morph)
    library_preds_val["Topo_D"] = preds_d

    # Verify predictions shape
    n_classes = len(data["classes"])
    assert preds_a.shape == (len(y_val), n_classes), "Prediction shape mismatch."

    # 3. Ensemble Selection
    print("Running Greedy Ensemble Selection...")
    selector = GreedyEnsemble(max_iterations=10, tol=1e-4)
    selector.fit(library_preds_val, y_val, verbose=True)

    # Verify selection
    assert (
        len(selector.selected_experts) > 0
    ), "Ensemble should select at least one expert."
    print(f"Selected Experts: {selector.selected_experts}")
    print(f"Best Validation Log Loss: {selector.best_score:.4f}")

    # 4. Generate Final Predictions (Simulated Test)
    print("Generating Ensemble Predictions...")
    # We reuse val predictions here just to demonstrate the predict method
    ensemble_preds = selector.predict(library_preds_val)

    assert ensemble_preds.shape == preds_a.shape, "Ensemble output shape mismatch."
    assert np.allclose(
        ensemble_preds.sum(axis=1), 1.0
    ), "Ensemble probabilities must sum to 1."

    print("Pipeline and Ensemble verification passed.")


if __name__ == "__main__":
    try:
        demo_utils()
        demo_image_features()
        data = demo_data_loading()
        demo_pipelines_and_ensemble(data)
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
