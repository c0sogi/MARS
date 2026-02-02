import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Import provided library modules
from library.config import Config
from library.transforms import get_stream_transforms
from library.dataset import get_dataset, DogDataset
from library.model_factory import load_backbone
from library.feature_manager import extract_features
from library.linear_probe import StreamClassifier
from library.ensemble_optimizer import (
    optimize_ensemble_weights,
    blend_predictions,
    generate_submission,
)


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting Dual-Stream Ensemble Demo...")
    set_seeds(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    # We override the LogisticRegressionCV parameters to ensure it runs fast
    # and works with our tiny debug dataset (cv=2 requires min 2 samples per class).
    Config.LOGREG_PARAMS = {
        "Cs": 2,  # Only try 2 regularization strengths
        "cv": 2,  # 2-fold cross-validation
        "max_iter": 50,  # Limit iterations
        "n_jobs": 1,  # No multiprocessing for small data
        "random_state": Config.SEED,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "multi_class": "multinomial",
    }

    # Define a small limit for feature extraction to keep runtime short
    DEBUG_LIMIT = 20
    print(f"Debug limit set to {DEBUG_LIMIT} samples per split.")

    # -------------------------------------------------------------------------
    # 2. Verify Transforms
    # -------------------------------------------------------------------------
    print("\n--- Verifying Transforms (Stream A) ---")
    transforms_a = get_stream_transforms(Config.STREAM_A, is_train=False)

    # Create a dummy image (RGB)
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    )

    # Apply transforms
    views = {k: t(dummy_img) for k, t in transforms_a.items()}

    # Check shapes
    # Global: Config.STREAM_A['view_global_size'] = 224
    assert views["global"].shape == (
        3,
        224,
        224,
    ), f"Global shape mismatch: {views['global'].shape}"
    # Standard: CenterCrop(224)
    assert views["standard"].shape == (
        3,
        224,
        224,
    ), f"Standard shape mismatch: {views['standard'].shape}"
    # Local: CenterCrop(224)
    assert views["local"].shape == (
        3,
        224,
        224,
    ), f"Local shape mismatch: {views['local'].shape}"
    print("Transforms verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset
    # -------------------------------------------------------------------------
    print("\n--- Verifying Dataset ---")
    # Load Train Dataset
    train_ds = get_dataset("train", Config.STREAM_A)
    print(f"Train dataset size: {len(train_ds)}")

    # Get one sample
    sample = train_ds[0]
    assert "id" in sample
    assert "views" in sample
    assert "label" in sample
    assert isinstance(sample["views"], dict)
    assert "global" in sample["views"]
    print(f"Sample ID: {sample['id']}, Label: {sample['label']}")
    print("Dataset verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Feature Extraction (Stream A & B)
    # -------------------------------------------------------------------------
    print("\n--- Running Feature Extraction (Debug Mode) ---")
    # We extract features for Train, Val, and Test for both streams.
    # We use debug_limit to process only a few images.

    splits = ["train", "val", "test"]
    feats = {"A": {}, "B": {}}

    for stream_key, stream_config in [("A", Config.STREAM_A), ("B", Config.STREAM_B)]:
        for split in splits:
            print(f"Processing {stream_config['name']} - {split}...")
            # Note: load_cached_data=False ensures we run the extraction logic
            emb, ids, lbl = extract_features(
                stream_config, split, load_cached_data=False, debug_limit=DEBUG_LIMIT
            )
            feats[stream_key][split] = (emb, ids, lbl)

            # Verify embedding shape
            # ConvNeXt-Large dim=1536, EVA02-Large dim=1024
            # We concatenate 3 views, so dim * 3
            expected_dim = 1536 * 3 if stream_key == "A" else 1024 * 3
            assert emb.shape == (DEBUG_LIMIT, expected_dim)
            assert len(ids) == DEBUG_LIMIT

    # -------------------------------------------------------------------------
    # 5. Linear Probe Training
    # -------------------------------------------------------------------------
    print("\n--- Training Linear Probes ---")

    # To ensure LogisticRegressionCV works with only 20 samples, we create
    # synthetic labels. Real labels might have 20 unique classes, causing CV failure.
    # We pick 2 real breed names and assign them 50/50.

    # Get real breed names from the loaded labels to be realistic
    real_labels_list = feats["A"]["train"][2]
    # Just pick two arbitrary strings if available, else hardcode
    if real_labels_list is not None and len(np.unique(real_labels_list)) >= 2:
        classes_to_use = np.unique(real_labels_list)[:2]
    else:
        classes_to_use = ["breed_1", "breed_2"]

    print(f"Using synthetic classes for demo training: {classes_to_use}")

    # Create synthetic label array
    half = DEBUG_LIMIT // 2
    synthetic_train_labels = np.array(
        [classes_to_use[0]] * half + [classes_to_use[1]] * (DEBUG_LIMIT - half)
    )

    # Train Classifiers
    classifiers = {}
    for stream_key, stream_config in [("A", Config.STREAM_A), ("B", Config.STREAM_B)]:
        clf = StreamClassifier(stream_config)

        # Train on extracted embeddings with synthetic labels
        train_emb = feats[stream_key]["train"][0]
        clf.train(train_emb, synthetic_train_labels)

        classifiers[stream_key] = clf

    # -------------------------------------------------------------------------
    # 6. Inference & Ensemble Optimization
    # -------------------------------------------------------------------------
    print("\n--- Optimizing Ensemble ---")

    # Predict on Validation set
    # We need synthetic validation labels matching the training classes to calc LogLoss
    synthetic_val_labels = np.array(
        [classes_to_use[0]] * half + [classes_to_use[1]] * (DEBUG_LIMIT - half)
    )

    val_preds = {}
    for stream_key in ["A", "B"]:
        val_emb = feats[stream_key]["val"][0]
        val_preds[stream_key] = classifiers[stream_key].predict(val_emb)

    # Optimize
    # Note: classifiers['A'].classes_ should match classes_to_use
    class_names = classifiers["A"].classes_
    weight_a = optimize_ensemble_weights(
        val_preds["A"], val_preds["B"], synthetic_val_labels, class_names
    )

    assert 0.0 <= weight_a <= 1.0
    print(f"Optimization successful. Weight A: {weight_a:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    # Predict on Test set
    test_preds = {}
    for stream_key in ["A", "B"]:
        test_emb = feats[stream_key]["test"][0]
        test_preds[stream_key] = classifiers[stream_key].predict(test_emb)

    # Blend
    blended_test_preds = blend_predictions(test_preds["A"], test_preds["B"], weight_a)

    # Get Test IDs
    test_ids = feats["A"]["test"][1]

    # Generate CSV
    # Note: This submission will only have columns for the 2 classes we trained on.
    # In a real run, it would have all 120 classes.
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    generate_submission(test_ids, blended_test_preds, class_names, submission_path)

    # Verify
    assert os.path.exists(submission_path)
    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Check if IDs match
    assert df_sub["id"].iloc[0] == test_ids[0]

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
