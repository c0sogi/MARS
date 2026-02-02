import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import from the provided library
from library.config import RANDOM_SEED
from library.utils import set_seed, extract_genus, get_species_to_genus_mapping
from library.data_loader import load_and_process_data
from library.model_definitions import (
    build_linear_species_model,
    build_generative_species_model,
    build_quadratic_species_model,
    build_genus_supervisor_model,
)
from library.hierarchical_engine import TaxonomyEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # 1. Setup and Reproducibility
    print("[1] Setting random seed...")
    set_seed(RANDOM_SEED)

    # 2. Data Loading
    print("[2] Loading and processing data...")
    # This function handles caching, scaling, and encoding
    X_train, X_test, y_species, y_genus, test_ids, scaler, species_le, genus_le = (
        load_and_process_data(load_cached_data=True)
    )

    # Validation of loaded data
    n_samples_train, n_features = X_train.shape
    n_species = len(species_le.classes_)
    n_genera = len(genus_le.classes_)

    print(f"    Train shape: {X_train.shape}")
    print(f"    Test shape: {X_test.shape}")
    print(f"    Number of Species: {n_species}")
    print(f"    Number of Genera: {n_genera}")

    assert n_features == 192, f"Expected 192 features, got {n_features}"
    assert len(y_species) == n_samples_train
    assert len(y_genus) == n_samples_train

    # 3. Utility Function Verification
    print("\n[3] Verifying Utility Functions...")

    # Test Genus Extraction
    sample_species = "Acer_Capillipes"
    extracted = extract_genus(sample_species)
    print(f"    extract_genus('{sample_species}') -> '{extracted}'")
    assert extracted == "Acer", "Genus extraction failed"

    # Test Mapping Generation
    mapping = get_species_to_genus_mapping(species_le, genus_le)
    print(f"    Mapping shape: {mapping.shape}")
    assert mapping.shape == (n_species,), "Mapping shape mismatch"
    # Verify a specific mapping (e.g., first species maps to a valid genus index)
    assert 0 <= mapping[0] < n_genera, "Mapping index out of bounds"

    # 4. Model Definitions & Optimization
    print("\n[4] Initializing and Optimizing Models...")

    # We instantiate the ensemble class
    ensemble = TaxonomyEnsemble()

    # OPTIMIZATION FOR DEMO SPEED:
    # The default models in TaxonomyEnsemble use high max_iter and large grids.
    # We will manually replace them with lightweight versions for this demonstration.

    fast_cv_folds = 2
    fast_max_iter = 10
    fast_grid = [1.0]  # Single regularization strength for speed

    print("    Replacing internal models with fast, lightweight versions...")

    # A. Linear Discriminative
    ensemble.linear_model = build_linear_species_model(
        max_iter=fast_max_iter, cv_folds=fast_cv_folds, cs_grid=fast_grid
    )

    # B. Generative (LDA is naturally fast, but we stick to defaults or small adjustments)
    ensemble.generative_model = build_generative_species_model()

    # C. Quadratic Discriminative
    ensemble.quadratic_model = build_quadratic_species_model(
        max_iter=fast_max_iter,
        cv_folds=fast_cv_folds,
        cs_grid=fast_grid,
        pca_variance=0.8,  # Retain less variance for speed
    )

    # D. Genus Supervisor
    ensemble.genus_model = build_genus_supervisor_model(
        max_iter=fast_max_iter, cv_folds=fast_cv_folds, cs_grid=fast_grid
    )

    # 5. Ensemble Execution (Training & Inference)
    print("\n[5] Executing Ensemble Pipeline...")

    # Use a small subset of data to demonstrate functionality quickly
    subset_size = 100
    X_sub = X_train[:subset_size]
    y_species_sub = y_species[:subset_size]
    y_genus_sub = y_genus[:subset_size]

    print(f"    Fitting on subset of {subset_size} samples...")
    ensemble.fit(X_sub, y_species_sub, y_genus_sub, species_le, genus_le)

    print("    Generating predictions on test subset...")
    X_test_sub = X_test[:10]
    probs = ensemble.predict_proba(X_test_sub)

    # 6. Output Validation
    print("\n[6] Validating Predictions...")
    print(f"    Prediction shape: {probs.shape}")

    # Check shape
    assert probs.shape == (
        10,
        n_species,
    ), f"Expected shape (10, {n_species}), got {probs.shape}"

    # Check probability constraints (rows sum to 1)
    row_sums = probs.sum(axis=1)
    print(f"    Row sums (first 5): {row_sums[:5]}")
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Check value range
    assert probs.min() >= 0 and probs.max() <= 1.0, "Probabilities out of [0, 1] range"

    # Calculate Score (Log Loss) on training subset
    score = ensemble.score(X_sub, y_species_sub)
    print(f"    Training Subset Log Loss: {score:.4f}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
