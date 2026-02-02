import os
import shutil
import numpy as np
import pandas as pd
from library import (
    utils,
    data_loader,
    feature_extraction,
    preprocessing,
    models,
    ensemble,
)


def run_demo():
    # 1. Setup and Utils Verification
    print("=== 1. Testing Utils ===")
    utils.set_seed(42)

    # Verify probability clipping
    probs = np.array([[0.0, 0.5, 1.0], [1e-20, 0.5, 1 - 1e-20]])
    clipped = utils.clip_probabilities(probs)
    assert (
        clipped.min() >= 1e-15
    ), "Probabilities were not clipped correctly (lower bound)"
    assert (
        clipped.max() <= 1 - 1e-15
    ), "Probabilities were not clipped correctly (upper bound)"
    print("Utils verified.")

    # 2. Data Loading
    print("\n=== 2. Testing Data Loader ===")
    # Load a small subset of training data to verify image loading logic
    subset_size = 10
    df_train_subset, images_subset = data_loader.get_data_split(
        "train", max_samples=subset_size
    )

    assert len(df_train_subset) == subset_size, f"Expected {subset_size} metadata rows"
    assert len(images_subset) == subset_size, f"Expected {subset_size} loaded images"
    assert (
        df_train_subset.shape[1] == 195
    ), "Metadata should have 195 columns (192 features + id + species + path)"
    print(f"Successfully loaded {subset_size} training samples.")

    # 3. Feature Extraction
    print("\n=== 3. Testing Feature Extraction ===")
    # We use the validation split (179 samples) for this test as it's smaller than train (712)
    # This function extracts Morphometrics from images and loads pre-extracted tabular features
    print("Generating/Loading feature views for validation set...")
    views_val = feature_extraction.get_feature_views("val", load_cached_data=True)

    # Verify structure
    assert "views" in views_val, "Result should contain 'views' dictionary"
    assert "ids" in views_val, "Result should contain 'ids'"
    assert "y" in views_val, "Result should contain 'y' (labels)"

    # Verify specific views
    # Global: 192 features (Margin + Shape + Texture)
    # Morphometrics: 11 features (Extracted from image)
    X_global_val = views_val["views"]["Global"]
    X_morph_val = views_val["views"]["Morphometrics"]

    assert X_global_val.shape == (
        179,
        192,
    ), f"Global view shape mismatch: {X_global_val.shape}"
    assert X_morph_val.shape == (
        179,
        11,
    ), f"Morphometrics view shape mismatch: {X_morph_val.shape}"
    print("Feature extraction verified.")

    # 4. Preprocessing
    print("\n=== 4. Testing Preprocessing ===")
    # Test a specific strategy: 'global_marginal' (PowerTransformer on Global view)
    # This function loads Train, Val, and Test, fits on Train, and transforms all.
    print("Applying 'global_marginal' preprocessing strategy...")
    data_proc = preprocessing.get_preprocessed_data(
        "global_marginal", load_cached_data=True
    )

    X_train_proc = data_proc["X_train"]
    y_train = data_proc["y_train"]
    X_val_proc = data_proc["X_val"]

    # Verify shapes
    # Train set size is 712
    assert X_train_proc.shape == (
        712,
        192,
    ), f"Processed Train shape mismatch: {X_train_proc.shape}"
    assert len(y_train) == 712, "Label count mismatch"
    assert X_val_proc.shape == (
        179,
        192,
    ), f"Processed Val shape mismatch: {X_val_proc.shape}"

    # Verify values are not all zero (transformation happened)
    assert np.abs(X_train_proc).mean() > 0, "Processed data contains all zeros"
    print("Preprocessing verified.")

    # 5. Models
    print("\n=== 5. Testing Models ===")
    # Instantiate and train an LDA model
    print("Training LDAWrapper...")
    lda = models.LDAWrapper(solver="lsqr", shrinkage=0.1, random_state=42)
    lda.fit(X_train_proc, y_train)

    # Predict on validation set
    val_preds = lda.predict_proba(X_val_proc)

    # Verify predictions
    n_classes = 99
    assert val_preds.shape == (
        179,
        n_classes,
    ), f"Prediction shape mismatch: {val_preds.shape}"
    # Check if rows sum to approx 1
    assert np.allclose(val_preds.sum(axis=1), 1.0), "Probabilities do not sum to 1"
    print("Model training and prediction verified.")

    # 6. Ensemble Pipeline
    print("\n=== 6. Testing Ensemble Pipeline ===")
    # Initialize the selector
    selector = ensemble.GreedyEnsembleSelector(random_state=42)

    # OPTIMIZATION FOR DEMO SPEED:
    # The default library has ~10 experts. We reduce this to 2 to make the selection phase fast.
    original_candidate_count = len(selector.candidates)
    selector.candidates = selector.candidates[:2]
    print(
        f"Reduced ensemble candidates from {original_candidate_count} to {len(selector.candidates)} for speed."
    )

    # Phase 1: Selection
    print("Running fit_selection (Greedy Forward Selection)...")
    selector.fit_selection(load_cached_data=True)

    # Ensure at least one expert was selected
    assert len(selector.selected_experts) > 0, "No experts were selected"
    print(f"Selected {len(selector.selected_experts)} experts.")

    # Phase 2: Refit
    print("Running refit_final (Retraining on Train+Val)...")
    selector.refit_final(load_cached_data=True)

    # Phase 3: Inference
    print("Generating submission predictions...")
    df_submission = selector.predict_submission(load_cached_data=True)

    # Verify Submission Format
    assert "id" in df_submission.columns, "Submission missing 'id' column"
    assert df_submission.shape == (
        99,
        100,
    ), f"Submission shape mismatch: {df_submission.shape}"  # 99 test samples, 1 id + 99 classes

    # Save dummy submission
    output_path = "./working/demo_submission.csv"
    df_submission.to_csv(output_path, index=False)
    print(f"Demo submission saved to {output_path}")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    # Ensure the working directory exists
    os.makedirs("./working", exist_ok=True)
    run_demo()
