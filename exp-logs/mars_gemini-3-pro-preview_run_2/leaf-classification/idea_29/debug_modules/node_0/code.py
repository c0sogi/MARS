import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Ensure the current directory is in the python path for library imports
sys.path.append(os.getcwd())

# Import provided library modules
from library.utils import set_seed, clipped_log_loss
from library.features import extract_morphometrics
from library.data import LeafDataManager
from library.models import HierarchicalEnsemble
from library.ensemble import GreedyForwardSelector

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Leaf Classification Library Demo ===\n")

    # 1. Setup
    print("Step 1: Setting random seed...")
    set_seed(42)

    # 2. Demonstrate Feature Extraction (library.features)
    print("\nStep 2: Testing Feature Extraction...")
    # Load metadata to get a valid image path
    train_meta_path = "./metadata/train.csv"
    if os.path.exists(train_meta_path):
        df_train = pd.read_csv(train_meta_path)
        # Get first image path
        first_image_rel_path = df_train.iloc[0]["image_path"]
        first_image_full_path = os.path.join("./input", first_image_rel_path)

        print(f"  Extracting morphometrics from: {first_image_rel_path}")
        features = extract_morphometrics(first_image_full_path)

        # Validation
        expected_keys = ["aspect_ratio", "solidity", "extent", "eccentricity"] + [
            f"hu_{i}" for i in range(7)
        ]
        for k in expected_keys:
            if k not in features:
                raise AssertionError(f"Missing feature key: {k}")

        print(f"  Successfully extracted {len(features)} morphometric features.")
        print(
            f"  Sample values: Solidity={features['solidity']:.4f}, Aspect Ratio={features['aspect_ratio']:.4f}"
        )
    else:
        raise FileNotFoundError(f"Metadata file not found at {train_meta_path}")

    # 3. Demonstrate Data Management (library.data)
    print("\nStep 3: Loading and Processing Data...")
    # Use a specific cache directory for this demo
    cache_dir = "./working/demo_cache"
    data_manager = LeafDataManager(metadata_dir="./metadata", cache_dir=cache_dir)

    # Force reload to demonstrate processing pipeline
    data = data_manager.load_data(load_cached_data=False)

    # Validation of Data Shapes
    # Train set size from metadata is 712
    n_train = 712
    n_val = 179
    n_test = 99

    # Check Global View (192 features: 64 margin + 64 shape + 64 texture)
    assert data["X_train_global"].shape == (
        n_train,
        192,
    ), f"X_train_global shape mismatch: {data['X_train_global'].shape}"
    assert data["X_val_global"].shape == (
        n_val,
        192,
    ), f"X_val_global shape mismatch: {data['X_val_global'].shape}"

    # Check Macro View (11 features: 7 Hu + 4 Scalars)
    # Note: features.py extracts 11 features.
    assert (
        data["X_train_macro"].shape[1] == 11
    ), f"X_train_macro feature count mismatch: {data['X_train_macro'].shape[1]}"

    # Check Combined View (192 + 11 = 203)
    assert (
        data["X_train_combined"].shape[1] == 203
    ), f"X_train_combined feature count mismatch: {data['X_train_combined'].shape[1]}"

    # Check Targets
    assert len(data["y_train"]) == n_train
    assert len(data["classes"]) == 99  # 99 species

    print(f"  Data loaded successfully.")
    print(f"  Train shape (Global): {data['X_train_global'].shape}")
    print(f"  Val shape (Global):   {data['X_val_global'].shape}")
    print(f"  Test shape (Global):  {data['X_test_global'].shape}")

    # 4. Demonstrate Hierarchical Ensemble (library.models)
    print("\nStep 4: Training Hierarchical Ensemble...")
    # Initialize ensemble with low iterations for speed
    ensemble = HierarchicalEnsemble(selection_iterations=3, random_seed=42)

    # Fit the ensemble (Train Candidates -> Select -> Retrain)
    ensemble.fit(data)

    # Predict on Test Data
    test_preds = ensemble.predict(data)

    # Validation of Predictions
    assert test_preds.shape == (
        n_test,
        99,
    ), f"Prediction shape mismatch: {test_preds.shape}"
    assert np.all(
        (test_preds >= 0) & (test_preds <= 1)
    ), "Predictions contain values outside [0, 1]"
    # Check if rows sum approximately to 1 (they might not exactly due to floating point, but close)
    # The metric function handles normalization, but raw probs should be reasonable.
    row_sums = test_preds.sum(axis=1)
    print(f"  Prediction mean row sum: {row_sums.mean():.4f}")

    print("  Ensemble training and prediction complete.")

    # 5. Demonstrate Standalone Greedy Selector (library.ensemble)
    print("\nStep 5: Demonstrating Standalone GreedyForwardSelector...")

    # Create two simple baseline models to act as candidates
    model_a = GaussianNB()
    model_b = LinearDiscriminantAnalysis()

    # Train on Global view
    model_a.fit(data["X_train_global"], data["y_train"])
    model_b.fit(data["X_train_global"], data["y_train"])

    # Generate Validation Predictions
    preds_a = model_a.predict_proba(data["X_val_global"])
    preds_b = model_b.predict_proba(data["X_val_global"])

    preds_dict = {"GaussianNB": preds_a, "LDA": preds_b}

    # Run Selector
    selector = GreedyForwardSelector(selection_iterations=5, random_seed=42)
    weights = selector.fit(preds_dict, data["y_val"])

    # Validation
    assert len(weights) > 0, "Selector returned no weights."
    total_weight = sum(weights.values())
    assert (
        total_weight == 5
    ), f"Total weight {total_weight} does not match selection iterations (5)."

    print(f"  Selected Weights: {weights}")
    print(f"  Best Validation Score: {selector.best_score:.4f}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
