import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path to import the library modules
sys.path.append(".")

from library import config, data_utils, feature_extractor, classifier


def run_demo():
    print("=== Starting End-to-End Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("1. Configuring environment...")

    # Define a specific working directory for this demo to avoid overwriting real work
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config paths to point to the demo directory
    config.WORKING_DIR = demo_working_dir
    config.CACHE_PATHS = {
        "train_embeddings": os.path.join(demo_working_dir, "train_embeddings.npy"),
        "train_labels": os.path.join(demo_working_dir, "train_labels.npy"),
        "val_embeddings": os.path.join(demo_working_dir, "val_embeddings.npy"),
        "val_labels": os.path.join(demo_working_dir, "val_labels.npy"),
        "test_embeddings": os.path.join(demo_working_dir, "test_embeddings.npy"),
        "test_ids": os.path.join(demo_working_dir, "test_ids.npy"),
        "model": os.path.join(demo_working_dir, "logreg_model.joblib"),
        "submission": os.path.join(demo_working_dir, "submission.csv"),
    }

    # Override Classifier Hyperparameters for Speed
    # We reduce Cross-Validation folds to 2 and max iterations to 50
    config.LOGREG_CV = 2
    config.LOGREG_MAX_ITER = 50

    # Set Random Seeds
    config.set_seed(42)
    print("   Configuration complete.\n")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("2. Demonstrating Data Loading...")

    # Initialize Dataset for Training with 'global' view
    # 'global' view returns 2 tensors per image (Original + Horizontal Flip)
    dataset = data_utils.get_dataset(split="train", view_type="global")

    # Fetch a single sample
    img_tensor, label, img_id = dataset[0]

    print(f"   Fetched Sample ID: {img_id}")
    print(f"   Label Index: {label}")
    print(f"   Tensor Shape: {img_tensor.shape}")

    # Validations
    assert isinstance(img_tensor, torch.Tensor), "Image data must be a torch.Tensor"
    # Expected shape: (2 views, 3 channels, 224 height, 224 width)
    assert img_tensor.shape == (
        2,
        3,
        224,
        224,
    ), f"Unexpected tensor shape: {img_tensor.shape}"
    assert isinstance(label, int), "Label must be an integer"
    assert isinstance(img_id, str), "ID must be a string"
    print("   Data Loading verification passed.\n")

    # ---------------------------------------------------------
    # 3. Feature Extraction Demonstration
    # ---------------------------------------------------------
    print("3. Demonstrating Feature Extraction...")

    # We use a very small subset (20 samples) for the demo to run instantly
    debug_size = 20

    # Extract Train Features
    print("   Extracting training features (subset)...")
    train_feats, train_lbls, train_ids = feature_extractor.get_features(
        split="train",
        view_type="global",
        load_cached_data=False,  # Force extraction
        debug_sample_size=debug_size,
    )

    # Extract Validation Features
    print("   Extracting validation features (subset)...")
    val_feats, val_lbls, val_ids = feature_extractor.get_features(
        split="val",
        view_type="global",
        load_cached_data=False,
        debug_sample_size=debug_size,
    )

    # Validations
    # ConvNeXt Large output dimension is 1536
    expected_dim = 1536
    assert train_feats.shape == (
        debug_size,
        expected_dim,
    ), f"Feature shape mismatch: {train_feats.shape}"
    assert len(train_lbls) == debug_size
    assert len(train_ids) == debug_size
    print("   Feature Extraction verification passed.\n")

    # ---------------------------------------------------------
    # 4. Classifier Training Demonstration
    # ---------------------------------------------------------
    print("4. Demonstrating Classifier Training...")

    # PREPROCESSING FOR DEMO:
    # LogisticRegressionCV requires at least 2 samples per class per fold.
    # With a random subset of 20 samples and 120 classes, classes will be sparse.
    # We create a synthetic dataset from the extracted features by filtering for
    # the top 2 classes and duplicating the data to ensure robust training.

    unique_classes, counts = np.unique(train_lbls, return_counts=True)
    # Ensure we have at least 2 classes
    if len(unique_classes) < 2:
        target_classes = [0, 1]
        # Mock labels if subset is too homogenous
        train_lbls[:10] = 0
        train_lbls[10:] = 1
    else:
        target_classes = unique_classes[:2]

    mask = np.isin(train_lbls, target_classes)
    demo_feats = train_feats[mask]
    demo_lbls = train_lbls[mask]

    # Tile (duplicate) data 4 times to ensure enough samples for CV
    demo_feats = np.tile(demo_feats, (4, 1))
    demo_lbls = np.tile(demo_lbls, 4)

    print(
        f"   Training on synthetic subset: {demo_feats.shape[0]} samples, {len(np.unique(demo_lbls))} classes"
    )

    # Train Model
    model = classifier.train_classifier(demo_feats, demo_lbls, load_cached_model=False)

    # Validations
    assert model is not None, "Classifier training returned None"
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"
    print("   Classifier Training verification passed.\n")

    # ---------------------------------------------------------
    # 5. Evaluation Demonstration
    # ---------------------------------------------------------
    print("5. Demonstrating Evaluation...")

    # Prepare validation data (filter to same classes as training)
    val_mask = np.isin(val_lbls, target_classes)

    # If validation set doesn't have these classes (possible in small random subset), mock it
    if val_mask.sum() == 0:
        demo_val_feats = demo_feats[:10]
        demo_val_lbls = demo_lbls[:10]
    else:
        demo_val_feats = val_feats[val_mask]
        demo_val_lbls = val_lbls[val_mask]

    loss = classifier.evaluate_model(model, demo_val_feats, demo_val_lbls)

    # Validations
    assert isinstance(loss, float), "Log loss should be a float"
    assert loss >= 0, "Log loss should be non-negative"
    print(f"   Validation Loss: {loss:.4f}")
    print("   Evaluation verification passed.\n")

    # ---------------------------------------------------------
    # 6. Submission Demonstration
    # ---------------------------------------------------------
    print("6. Demonstrating Submission Generation...")

    # Extract Test Features
    print("   Extracting test features (subset)...")
    test_feats, _, test_ids = feature_extractor.get_features(
        split="test",
        view_type="global",
        load_cached_data=False,
        debug_sample_size=debug_size,
    )

    # Predict Probabilities
    # Note: The model is trained on a subset of classes (2 classes).
    # We must map these predictions to the full 120-class format required for submission.
    subset_probas = classifier.predict_proba(model, test_feats)

    # Initialize full probability matrix (N_samples, 120_classes)
    full_probas = np.zeros((len(test_feats), config.NUM_CLASSES))

    # Map predictions to correct columns
    # model.classes_ holds the class indices the model was trained on
    for i, class_idx in enumerate(model.classes_):
        full_probas[:, class_idx] = subset_probas[:, i]

    # Get list of all breed names for the header
    _, idx_to_class = data_utils.get_class_mapping()
    all_class_names = [idx_to_class[i] for i in range(config.NUM_CLASSES)]

    # Create Submission File
    classifier.create_submission(
        ids=test_ids,
        probabilities=full_probas,
        class_names=all_class_names,
        output_path=config.CACHE_PATHS["submission"],
    )

    # Validations
    assert os.path.exists(
        config.CACHE_PATHS["submission"]
    ), "Submission file was not created"

    df_sub = pd.read_csv(config.CACHE_PATHS["submission"])
    expected_cols = config.NUM_CLASSES + 1  # 120 breeds + 'id'
    assert df_sub.shape == (
        debug_size,
        expected_cols,
    ), f"Submission shape mismatch. Expected ({debug_size}, {expected_cols}), got {df_sub.shape}"

    print("   Submission Generation verification passed.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
