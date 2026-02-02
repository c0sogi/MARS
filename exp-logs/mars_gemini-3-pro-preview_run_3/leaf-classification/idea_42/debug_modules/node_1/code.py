import os
import numpy as np
import pandas as pd
import joblib
import shutil
from sklearn.pipeline import Pipeline

# Import provided library modules
from library.utils import seed_everything, WORKING_DIR, load_metadata
from library.feature_extractor import FeatureExtractor
from library.densification import Densifier
from library.model_factory import create_pipeline
from library.trainer import CrossValidator
from library.inference import Predictor


def main():
    # 1. Setup and Configuration
    # ==========================
    print("Initializing Demo...")
    seed_everything(42)

    # We use a small limit to ensure the demo runs quickly within the time constraints.
    # 6 samples allows for checking batch processing logic and reshaping.
    DEMO_LIMIT = 6
    DEMO_CACHE_SUBDIR = "demo_execution"

    # Clean up previous demo run if exists to ensure fresh execution
    demo_dir = os.path.join(WORKING_DIR, DEMO_CACHE_SUBDIR)
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    # 2. Feature Extractor Demonstration
    # ==================================
    print("\n--- Testing FeatureExtractor ---")
    extractor = FeatureExtractor()

    # Extract features for a small subset of training data
    # We force load_cached_data=False to demonstrate actual extraction logic
    dino_feats, conv_feats, tab_feats, ids = extractor.extract_and_save_features(
        split="train", load_cached_data=False, limit=DEMO_LIMIT
    )

    # Verify shapes
    # Expected: (N, 12, 1024) for DINO, (N, 12, 1536) for ConvNeXt, (N, 192) for Tabular
    print(f"Extracted DINO shape: {dino_feats.shape}")
    print(f"Extracted ConvNeXt shape: {conv_feats.shape}")

    assert dino_feats.shape == (DEMO_LIMIT, 12, 1024), "DINOv2 feature shape mismatch"
    assert conv_feats.shape == (DEMO_LIMIT, 12, 1536), "ConvNeXt feature shape mismatch"
    assert tab_feats.shape == (DEMO_LIMIT, 192), "Tabular feature shape mismatch"
    assert ids.shape == (DEMO_LIMIT,), "IDs shape mismatch"
    print("Feature Extraction verification passed.")

    # 3. Densification Demonstration
    # ==============================
    print("\n--- Testing Densifier ---")
    densifier = Densifier(cache_subdir=DEMO_CACHE_SUBDIR)

    # Create dummy labels for the densification test since extractor doesn't return them
    dummy_labels = np.random.randint(0, 5, size=DEMO_LIMIT)

    # Test Training Densification (Convex-Hull, 6x expansion)
    dino_dense, conv_dense, tab_dense, y_dense, ids_dense = (
        densifier.densify_training_data(
            dino_feats,
            conv_feats,
            tab_feats,
            dummy_labels,
            ids,
            split_name="demo_train",
            load_cached_data=False,
        )
    )

    # Verify shapes: Should be N * 6
    expected_rows = DEMO_LIMIT * 6
    print(f"Densified Training Rows: {dino_dense.shape[0]} (Expected: {expected_rows})")
    assert dino_dense.shape[0] == expected_rows
    assert conv_dense.shape[0] == expected_rows
    assert y_dense.shape[0] == expected_rows

    # Test Inference Densification (Canonical, 3x expansion)
    dino_canon, conv_canon, tab_canon, ids_canon = densifier.densify_inference_data(
        dino_feats,
        conv_feats,
        tab_feats,
        ids,
        split_name="demo_inf",
        load_cached_data=False,
    )

    # Verify shapes: Should be N * 3
    expected_inf_rows = DEMO_LIMIT * 3
    print(
        f"Canonical Inference Rows: {dino_canon.shape[0]} (Expected: {expected_inf_rows})"
    )
    assert dino_canon.shape[0] == expected_inf_rows
    print("Densification verification passed.")

    # 4. Model Pipeline Demonstration
    # ===============================
    print("\n--- Testing Model Pipeline ---")
    pipeline = create_pipeline(dino_dim=1024, conv_dim=1536, tab_dim=192)

    # Construct input matrix [DINO | Conv | Tab]
    X_train = np.hstack([dino_dense, conv_dense, tab_dense])

    # Fit pipeline
    print("Fitting pipeline on densified data...")
    pipeline.fit(X_train, y_dense)

    # Predict
    X_inf = np.hstack([dino_canon, conv_canon, tab_canon])
    probs = pipeline.predict_proba(X_inf)

    print(f"Prediction shape: {probs.shape}")
    # Output classes depends on the unique values in dummy_labels
    n_classes_demo = len(np.unique(dummy_labels))
    # Note: LDA might infer classes based on input y, so shape[1] should match unique classes in y_dense
    assert probs.shape[0] == expected_inf_rows
    print("Model Pipeline verification passed.")

    # 5. Full Cross-Validation Workflow
    # =================================
    print("\n--- Testing CrossValidator ---")
    # Initialize CV with 2 splits for speed
    cv = CrossValidator(n_splits=2, random_state=42, cache_subdir=DEMO_CACHE_SUBDIR)

    # Run CV (this handles extraction, densification, and training internally)
    # We pass limit=DEMO_LIMIT to use the small subset
    avg_score = cv.run_cv(load_cached_data=False, limit=DEMO_LIMIT)

    print(f"CV Finished. Average Score: {avg_score}")

    # Verify models were saved
    models_dir = os.path.join(WORKING_DIR, DEMO_CACHE_SUBDIR, "models")
    assert os.path.exists(os.path.join(models_dir, "pipeline_fold_0.pkl"))
    assert os.path.exists(os.path.join(models_dir, "pipeline_fold_1.pkl"))
    assert os.path.exists(os.path.join(models_dir, "classes.pkl"))
    print("Cross-Validation verification passed.")

    # 6. Inference and Submission Generation
    # ======================================
    print("\n--- Testing Predictor ---")
    predictor = Predictor(cache_subdir=DEMO_CACHE_SUBDIR)

    # Generate submission for the test set (limited)
    predictor.generate_submission(load_cached_data=False, limit=DEMO_LIMIT)

    # Verify submission file
    submission_path = os.path.join(WORKING_DIR, "submission", "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check structure
    assert "id" in df_sub.columns
    assert len(df_sub) == DEMO_LIMIT

    # Check probability constraints (sum to 1, range 0-1) are handled by save_submission
    # We just check one row sum approx 1
    row_sums = df_sub.drop(columns=["id"]).sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
