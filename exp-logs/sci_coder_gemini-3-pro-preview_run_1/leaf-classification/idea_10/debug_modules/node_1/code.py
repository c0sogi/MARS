import os
import numpy as np
import pandas as pd
import shutil
from library import config, features, dataset, transforms, classifier, trainer

# Ensure reproducibility
np.random.seed(42)


def clean_working_dir():
    """Clean up working directory to ensure fresh execution for the demo."""
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)


def demo_feature_extraction():
    print("\n--- Demo: Feature Extraction ---")

    # 1. Get a sample image path from metadata
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    sample_row = df_train.iloc[0]
    image_rel_path = sample_row["file_path"]
    full_image_path = os.path.join(config.INPUT_DIR, image_rel_path)

    print(f"Testing feature extraction on: {full_image_path}")

    # 2. Extract geometric properties
    props = features.extract_geometric_props(full_image_path)
    print(f"Extracted properties: {props}")

    # 3. Validation
    assert isinstance(props, dict), "Output must be a dictionary"
    assert "area" in props, "Dictionary must contain 'area'"
    assert "aspect_ratio" in props, "Dictionary must contain 'aspect_ratio'"
    assert props["area"] >= 0, "Area must be non-negative"
    assert props["aspect_ratio"] >= 0, "Aspect ratio must be non-negative"
    print("Feature extraction logic verified.")


def demo_dataset_loading():
    print("\n--- Demo: Dataset Loading (Debug Mode) ---")

    # 1. Load data using the dataset module
    # Using debug_mode=True to load only a small subset (e.g., 20 rows)
    debug_size = 20
    train_data, val_data, test_data = dataset.load_dataset(
        load_cached_data=False,  # Force re-computation for demo
        debug_mode=True,
        debug_size=debug_size,
    )

    X_train, y_train, train_ids = train_data
    X_test, test_ids = test_data

    print(f"Loaded Train X shape: {X_train.shape}")
    print(f"Loaded Train y shape: {y_train.shape}")
    print(f"Loaded Test X shape: {X_test.shape}")

    # 2. Validation
    # Original features (192) + Augmented features (2) = 194
    expected_cols = 194
    assert X_train.shape == (
        debug_size,
        expected_cols,
    ), f"Expected ({debug_size}, {expected_cols})"
    assert len(y_train) == debug_size
    assert len(X_test) == debug_size
    assert "area" in X_train.columns, "Augmented feature 'area' missing"
    assert "aspect_ratio" in X_train.columns, "Augmented feature 'aspect_ratio' missing"

    print("Dataset loading and augmentation verified.")
    return X_train, y_train, X_test


def demo_preprocessing(X_train, X_test):
    print("\n--- Demo: Preprocessing Pipeline ---")

    # 1. Get Pipeline
    pipeline = transforms.get_pipeline()

    # 2. Fit and Transform
    print("Fitting pipeline on training data...")
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print(f"Transformed Train shape: {X_train_transformed.shape}")

    # 3. Validation
    assert X_train_transformed.shape == X_train.shape
    assert not np.isnan(X_train_transformed).any(), "Transformed data contains NaNs"

    # Check standardization (Mean ~ 0, Std ~ 1)
    means = np.mean(X_train_transformed, axis=0)
    stds = np.std(X_train_transformed, axis=0)
    print(f"Average Feature Mean: {np.mean(means):.4f}")
    print(f"Average Feature Std: {np.mean(stds):.4f}")

    # Allow some tolerance for small sample sizes
    assert np.abs(np.mean(means)) < 0.5, "Data not centered correctly"

    print("Preprocessing pipeline verified.")
    return X_train_transformed, X_test_transformed


def demo_classifier(X_train, y_train, X_test):
    print("\n--- Demo: Classifier (LeafLDA) ---")

    # 1. Initialize
    model = classifier.LeafLDA()

    # 2. Standard Fit
    print("Testing standard fit...")
    model.fit(X_train, y_train)

    # 3. Prediction
    probs = model.predict_proba(X_test)
    print(f"Prediction probabilities shape: {probs.shape}")

    assert probs.shape[0] == len(X_test)
    assert probs.shape[1] == len(model.classes_)
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"

    # 4. Transductive Fit
    print("Testing transductive fit...")
    # Use a low threshold to ensure at least some pseudo-labels are generated during this small demo
    model.fit_transductive(X_train, y_train, X_test, pseudo_label_threshold=0.1)

    print("Classifier logic verified.")


def demo_full_training_workflow():
    print("\n--- Demo: Full Training Workflow ---")

    # Execute the trainer module's main function
    # This integrates loading, transforming, training, and submission generation
    debug_size = 30
    trainer.train_transductive(
        load_cached_data=False,
        debug_mode=True,
        debug_size=debug_size,
        pseudo_label_threshold=0.90,
    )

    # Validation of the output
    submission_path = config.SUBMISSION_FILE_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created at {submission_path}")
    print(f"Submission shape: {df_sub.shape}")

    assert df_sub.shape[0] == debug_size, f"Expected {debug_size} rows in submission"
    assert "id" in df_sub.columns, "Submission missing 'id' column"

    # Check probability constraints
    # Drop ID column for check
    probs = df_sub.drop(columns=["id"]).values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    print("Full training workflow verified.")


if __name__ == "__main__":
    try:
        clean_working_dir()

        # Run individual component demos
        demo_feature_extraction()
        X_train, y_train, X_test = demo_dataset_loading()
        X_train_trans, X_test_trans = demo_preprocessing(X_train, X_test)
        demo_classifier(X_train_trans, y_train, X_test_trans)

        # Run integration demo
        demo_full_training_workflow()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
