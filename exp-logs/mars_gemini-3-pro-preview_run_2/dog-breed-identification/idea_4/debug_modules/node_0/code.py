import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_extractor as feature_extractor
import library.ensemble_logic as ensemble_logic


def main():
    print("Starting Dog Breed Prediction Pipeline Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # We use the full dataset (DEBUG=False) to ensure all 120 classes are present.
    # The A100 GPU is fast enough to process the full dataset (~10k images) quickly.
    config.DEBUG = False
    config.BATCH_SIZE = 64
    config.NUM_WORKERS = 4

    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    print(f"Configuration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Debug Mode: {config.DEBUG}")
    print(f"  Working Directory: {config.WORKING_DIR}")

    # Initialize storage for ensemble
    views = ["standard", "global", "local"]
    val_probs_dict = {}
    test_probs_dict = {}

    # We need to store ground truth / IDs to align things later
    y_val_true = None
    test_ids = None
    class_names = None

    # Load the backbone model once.
    print("\nLoading backbone model (ConvNeXt Large)...")
    backbone_model = feature_extractor.load_backbone(config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Process Each View (Feature Extraction & Individual Modeling)
    # -------------------------------------------------------------------------
    for view in views:
        print(f"\n{'='*40}")
        print(f"Processing View: {view}")
        print(f"{'='*40}")

        # A. Create DataLoaders
        # ---------------------
        print(f"Creating DataLoaders for view '{view}'...")
        train_loader, val_loader, test_loader, classes = data_loader.create_loaders(
            view
        )

        # Store class names for submission header (only need to do this once)
        if class_names is None:
            class_names = classes

        # Verify DataLoader output shape
        # Fetch one batch to check dimensions
        sample_imgs, _ = next(iter(train_loader))
        print(f"  Input tensor shape: {sample_imgs.shape}")
        # All views should output 224x224 due to the transform pipeline logic
        assert sample_imgs.shape[2:] == (
            224,
            224,
        ), f"Expected image size (224, 224), got {sample_imgs.shape[2:]}"

        # B. Extract Embeddings
        # ---------------------
        # We pass load_cached_data=False to force execution of the extraction logic for this demo.

        # Train
        print(f"  Extracting Train embeddings...")
        X_train, y_train = feature_extractor.get_embeddings(
            view, "train", train_loader, load_cached_data=False, model=backbone_model
        )

        # Validation
        print(f"  Extracting Validation embeddings...")
        X_val, y_val = feature_extractor.get_embeddings(
            view, "val", val_loader, load_cached_data=False, model=backbone_model
        )

        # Test
        print(f"  Extracting Test embeddings...")
        X_test, ids_test = feature_extractor.get_embeddings(
            view, "test", test_loader, load_cached_data=False, model=backbone_model
        )

        # Verify Embedding Shapes
        # ConvNeXt Large feature dim is 1536
        assert (
            X_train.shape[1] == 1536
        ), f"Expected 1536 features, got {X_train.shape[1]}"

        # Store/Verify Metadata consistency
        if y_val_true is None:
            y_val_true = y_val
            test_ids = ids_test
        else:
            # Ensure that the order of samples is preserved across views
            assert np.array_equal(
                y_val, y_val_true
            ), "Validation label mismatch between views!"
            assert np.array_equal(ids_test, test_ids), "Test ID mismatch between views!"

        # C. Train Logistic Regression
        # ----------------------------
        print(f"  Training Logistic Regression for {view}...")
        clf, val_probs, val_loss = ensemble_logic.train_logistic_regression(
            X_train, y_train, X_val, y_val, view_name=view
        )

        # Store validation predictions for ensemble optimization
        val_probs_dict[view] = val_probs

        # Predict on Test Set
        print(f"  Predicting on Test set...")
        test_probs = clf.predict_proba(X_test)
        test_probs_dict[view] = test_probs

        # Verify prediction shape (N_samples, N_classes)
        assert test_probs.shape[1] == len(
            classes
        ), f"Prediction classes {test_probs.shape[1]} != {len(classes)}"

    # -------------------------------------------------------------------------
    # 3. Ensemble Optimization
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("Optimizing Ensemble Weights")
    print(f"{'='*40}")

    # Prepare list of validation predictions ordered by view
    val_preds_list = [val_probs_dict[v] for v in views]

    # Calculate optimal weights
    weights = ensemble_logic.optimize_ensemble_weights(val_preds_list, y_val_true)

    # Verify weights sum to 1
    assert np.isclose(
        np.sum(weights), 1.0
    ), f"Weights do not sum to 1: {np.sum(weights)}"

    # -------------------------------------------------------------------------
    # 4. Generate Submission
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("Generating Final Submission")
    print(f"{'='*40}")

    # Prepare list of test predictions ordered by view (same order as weights)
    test_preds_list = [test_probs_dict[v] for v in views]

    # Compute weighted average
    final_test_probs = ensemble_logic.weighted_average_prediction(
        test_preds_list, weights
    )

    # Create Submission DataFrame
    # Format: id, breed1, breed2, ...
    submission_df = pd.DataFrame(final_test_probs, columns=class_names)

    # Add 'id' column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Verify Submission Dimensions
    # Rows = N_test, Cols = 1 (id) + 120 (breeds)
    n_test_samples = len(test_ids)
    n_classes = len(class_names)
    expected_shape = (n_test_samples, n_classes + 1)

    assert (
        submission_df.shape == expected_shape
    ), f"Submission shape mismatch. Expected {expected_shape}, got {submission_df.shape}"

    # Save to disk
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    print(f"Submission saved successfully to: {config.SUBMISSION_PATH}")
    print("Head of submission file:")
    print(submission_df.head())

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    main()
