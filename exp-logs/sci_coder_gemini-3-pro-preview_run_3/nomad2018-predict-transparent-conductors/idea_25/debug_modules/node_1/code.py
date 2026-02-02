import os
import sys
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import set_seed, SUBMISSION_PATH, CACHE_DIR
from library.data_loader import load_metadata, read_geometry
from library.feature_extractor import GeometricDescriptor, extract_features
from library.preprocessor import (
    TargetTransformer,
    FeatureCleaner,
    get_preprocessed_data,
)
from library.regressor import EnergyModel, train_and_generate_submission


def run_demo():
    print("Starting demonstration of library components...")
    set_seed(42)

    # Cite debug_lesson_13: Manually Invalidate Persistent Caches After Logic Fixes
    if os.path.exists(CACHE_DIR):
        print(f"Clearing cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loader ---")
    # Load a small subset of training metadata
    train_meta_small = load_metadata(split="train", max_rows=20)
    print(f"Loaded small train metadata shape: {train_meta_small.shape}")

    assert not train_meta_small.empty, "Metadata DataFrame should not be empty"
    assert "file_path" in train_meta_small.columns, "Metadata must contain file_path"
    assert (
        "formation_energy_ev_natom" in train_meta_small.columns
    ), "Metadata must contain targets"

    # Test reading geometry
    sample_path = train_meta_small.iloc[0]["file_path"]
    atoms = read_geometry(sample_path)
    print(f"Read geometry from {sample_path}: {atoms}")
    assert len(atoms) > 0, "Atoms object should contain atoms"

    # -------------------------------------------------------------------------
    # 2. Feature Extraction
    # -------------------------------------------------------------------------
    print("\n--- Testing Feature Extractor ---")
    descriptor = GeometricDescriptor()

    # Compute features for a single atoms object
    feats = descriptor.compute_global_features(atoms)
    feats.update(descriptor.compute_rdf_features(atoms))
    feats.update(descriptor.compute_interaction_features(atoms))
    feats.update(descriptor.compute_site_features(atoms))

    print(f"Extracted {len(feats)} features for a single structure.")
    assert "global_volume" in feats, "Global features missing"
    assert any(k.startswith("rdf_") for k in feats.keys()), "RDF features missing"

    # Test batch extraction with caching mechanism using a custom split name to avoid overwriting/loading real cache
    # We use the small metadata dataframe
    demo_split_name = "demo_train_subset"
    # Ensure we start fresh for this demo split
    demo_cache_path = os.path.join(CACHE_DIR, f"{demo_split_name}_features.parquet")
    if os.path.exists(demo_cache_path):
        os.remove(demo_cache_path)

    features_df = extract_features(
        train_meta_small, demo_split_name, load_cached_data=False
    )
    print(f"Batch extracted features shape: {features_df.shape}")
    assert len(features_df) == len(train_meta_small), "Feature DF row count mismatch"
    assert "id" in features_df.columns, "ID column missing in features DF"

    # -------------------------------------------------------------------------
    # 3. Preprocessing
    # -------------------------------------------------------------------------
    print("\n--- Testing Preprocessor ---")

    # Target Transformer
    tt = TargetTransformer()
    original_targets = train_meta_small["formation_energy_ev_natom"].values
    transformed = tt.transform(original_targets)
    inverted = tt.inverse_transform(transformed)
    print(
        f"Target Transformer check: Original={original_targets[0]:.4f}, Transformed={transformed[0]:.4f}, Inverted={inverted[0]:.4f}"
    )
    assert np.allclose(
        original_targets, inverted
    ), "Target transformation is not reversible"

    # Feature Cleaner
    # Separate features from metadata/targets
    exclude_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "file_path"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    cleaner = FeatureCleaner()
    cleaner.fit(features_df[feature_cols])
    cleaned_feats = cleaner.transform(features_df[feature_cols])

    print(f"Cleaned features shape: {cleaned_feats.shape}")
    # Check that constant columns (if any) might have been dropped, or at least NaNs filled
    assert not cleaned_feats.isnull().values.any(), "Cleaned features contain NaNs"

    # -------------------------------------------------------------------------
    # 4. Regression Model
    # -------------------------------------------------------------------------
    print("\n--- Testing Regressor ---")
    # Use very small hyperparameters for speed
    model = EnergyModel(n_estimators=5, max_depth=3, learning_rate=0.1)

    X = cleaned_feats
    y = train_meta_small[["formation_energy_ev_natom", "bandgap_energy_ev"]]

    # Fit model
    model.fit(X, y)
    print("Model fitted successfully.")

    # Predict
    preds = model.predict(X)
    print("Predictions generated.")
    print(preds.head())

    assert preds.shape == (len(X), 2), "Prediction shape mismatch"
    assert (
        preds.values >= 0
    ).all(), "Predictions should be non-negative (physical constraint)"

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n--- Testing Full Pipeline (Debug Mode) ---")
    # This runs the train_and_generate_submission function with debug=True
    # It uses a small subset of the real data (head(500)) and runs the full flow.
    # We override n_estimators to be very small to ensure it finishes quickly.

    train_and_generate_submission(debug=True, n_estimators=10)

    if os.path.exists(SUBMISSION_PATH):
        print(f"Submission file created at {SUBMISSION_PATH}")
        sub_df = pd.read_csv(SUBMISSION_PATH)
        print(f"Submission shape: {sub_df.shape}")
        # In debug mode, test_df is also processed.
        # The function loads test data. Since debug=True usually subsamples train/val,
        # but not explicitly test in the provided library code (it loads full test set),
        # submission should have rows equal to full test set (240).
        assert len(sub_df) == 240, f"Expected 240 predictions, got {len(sub_df)}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
