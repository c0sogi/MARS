import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config
from library import utils
from library import feature_engine
from library import logistic_model


def main():
    # 1. Setup and Configuration
    print("--- Step 1: Setup ---")
    utils.seed_everything(config.SEED)

    # Define a working directory for this demonstration
    demo_dir = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Define paths for temporary embedding files
    train_emb_path = os.path.join(demo_dir, "train_emb.npy")
    train_lbl_path = os.path.join(demo_dir, "train_lbl.npy")
    train_ids_path = os.path.join(demo_dir, "train_ids.npy")

    val_emb_path = os.path.join(demo_dir, "val_emb.npy")
    val_lbl_path = os.path.join(demo_dir, "val_lbl.npy")
    val_ids_path = os.path.join(demo_dir, "val_ids.npy")

    test_emb_path = os.path.join(demo_dir, "test_emb.npy")
    test_lbl_path = os.path.join(demo_dir, "test_lbl.npy")  # Dummy labels for test
    test_ids_path = os.path.join(demo_dir, "test_ids.npy")

    submission_path = os.path.join(demo_dir, "submission.csv")

    # Set debug limit to process only a small number of images for speed
    DEBUG_LIMIT = 50
    print(f"Debug mode enabled: Processing {DEBUG_LIMIT} images per dataset.")

    # 2. Feature Extraction
    print("\n--- Step 2: Feature Extraction ---")

    # Extract Train Features
    print("Processing Training Set...")
    train_X, train_y, train_ids = feature_engine.extract_embeddings(
        metadata_path=config.TRAIN_METADATA_PATH,
        embedding_path=train_emb_path,
        label_path=train_lbl_path,
        id_path=train_ids_path,
        load_cached_data=False,  # Force re-computation for demonstration
        debug=DEBUG_LIMIT,
    )

    # Verify Train Outputs
    assert (
        len(train_X) == DEBUG_LIMIT
    ), f"Expected {DEBUG_LIMIT} training embeddings, got {len(train_X)}"
    assert (
        len(train_y) == DEBUG_LIMIT
    ), f"Expected {DEBUG_LIMIT} training labels, got {len(train_y)}"
    assert train_X.ndim == 2, "Training embeddings should be 2D numpy array"
    # Feature dimension check: ConvNeXt Large (1536) + ViT Large (1024) = 2560 usually,
    # but exact dims depend on specific model variants in config.
    # We just ensure it's > 0.
    assert train_X.shape[1] > 0, "Embedding dimension must be positive"

    # Extract Validation Features
    print("Processing Validation Set...")
    val_X, val_y, val_ids = feature_engine.extract_embeddings(
        metadata_path=config.VAL_METADATA_PATH,
        embedding_path=val_emb_path,
        label_path=val_lbl_path,
        id_path=val_ids_path,
        load_cached_data=False,
        debug=DEBUG_LIMIT,
    )
    assert len(val_X) == DEBUG_LIMIT

    # Extract Test Features
    print("Processing Test Set...")
    test_X, test_y, test_ids = feature_engine.extract_embeddings(
        metadata_path=config.TEST_METADATA_PATH,
        embedding_path=test_emb_path,
        label_path=test_lbl_path,
        id_path=test_ids_path,
        load_cached_data=False,
        debug=DEBUG_LIMIT,
    )
    assert len(test_X) == DEBUG_LIMIT
    # Test labels should be -1 (dummy) or consistent with dataset logic
    # The dataset logic returns -1 if no label is found, or if it's the test set.

    # 3. Model Training
    print("\n--- Step 3: Model Training ---")

    # Train Logistic Regression
    # We use a smaller max_iter to ensure speed on this small debug subset
    clf = logistic_model.train_classifier(
        train_embeddings_path=train_emb_path,
        train_labels_path=train_lbl_path,
        val_embeddings_path=val_emb_path,
        val_labels_path=val_lbl_path,
        C=1.0,
        max_iter=100,
        seed=config.SEED,
    )

    # Verify Model
    assert clf is not None, "Classifier training failed (returned None)"
    assert hasattr(clf, "predict_proba"), "Classifier must have predict_proba method"
    # Check that model learned 120 classes (even if not all are present in the debug subset,
    # sklearn might only learn the ones present. However, for a robust pipeline,
    # we usually expect the full range. In this debug run with 50 samples,
    # it will likely only learn a subset of classes.
    # We will skip the class count assertion for this specific debug run
    # to avoid artificial failure, but in a real run, it should be 120.)
    print(f"Model classes found: {len(clf.classes_)}")

    # 4. Inference
    print("\n--- Step 4: Inference ---")

    probs = logistic_model.predict_probabilities(clf, test_emb_path)

    # Verify Probabilities
    assert probs.shape[0] == DEBUG_LIMIT, f"Expected {DEBUG_LIMIT} predictions"
    # The number of columns corresponds to the number of classes the model saw during training.
    # In a full run, this is 120. In debug, it's <= 50.
    assert probs.shape[1] == len(
        clf.classes_
    ), "Probability columns must match model classes"

    # 5. Submission Generation
    print("\n--- Step 5: Submission Generation ---")

    # Note: generate_submission assumes the model outputs probabilities for ALL 120 classes
    # in the correct order. Since our debug model might have fewer classes,
    # we need to pad the probabilities to match the expected 120 classes for the submission format check.
    # This is a specific adjustment for this debug demonstration.

    full_probs = np.zeros((DEBUG_LIMIT, config.NUM_CLASSES))

    # Map the predicted columns back to their indices in the full 120-class list
    # We need the class mapping to do this correctly.
    # We can get the idx_to_class from the dataset utility.
    class_to_idx, idx_to_class = dataset.get_class_mappings(config.TRAIN_METADATA_PATH)

    # clf.classes_ contains the class indices (integers) that were present in the training subset
    for i, class_idx in enumerate(clf.classes_):
        if class_idx < config.NUM_CLASSES:
            full_probs[:, class_idx] = probs[:, i]

    # Now generate the submission using the padded probabilities
    logistic_model.generate_submission(
        test_ids_path=test_ids_path,
        probabilities=full_probs,
        output_path=submission_path,
        train_metadata_path=config.TRAIN_METADATA_PATH,
    )

    # 6. Verification
    print("\n--- Step 6: Final Verification ---")

    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check rows
    assert len(df_sub) == DEBUG_LIMIT, f"Submission should have {DEBUG_LIMIT} rows"

    # Check columns: id + 120 breeds
    expected_cols = 1 + config.NUM_CLASSES
    assert (
        len(df_sub.columns) == expected_cols
    ), f"Submission should have {expected_cols} columns"

    # Check ID column
    assert "id" in df_sub.columns, "Submission must have 'id' column"
    assert df_sub["id"].iloc[0] == test_ids[0], "ID mismatch in submission"

    # Check probability sum (should be close to 1.0 for rows where we had predictions,
    # but since we padded with zeros for missing classes in debug, sums might be < 1.
    # If the model saw all classes, sums would be 1.
    # We just check that values are valid probabilities.)
    assert (df_sub.iloc[:, 1:].values >= 0).all(), "Probabilities must be non-negative"
    assert (df_sub.iloc[:, 1:].values <= 1).all(), "Probabilities must be <= 1"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
