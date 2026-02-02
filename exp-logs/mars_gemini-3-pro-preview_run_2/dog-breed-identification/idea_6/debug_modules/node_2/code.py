import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import Counter

# Import provided library components
from library.config import Config
from library.feature_engine import extract_features, get_class_to_idx
from library.modeling import (
    train_stream_classifier,
    predict_stream,
    evaluate_model,
    create_submission,
)
from library.ensemble import optimize_ensemble_weights, apply_ensemble


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_probabilities(probs, model_classes, total_classes=120):
    """
    Pads the probability matrix to shape (N, total_classes).
    Maps columns from model_classes to their correct indices.
    """
    if probs.shape[1] == total_classes:
        return probs

    N = probs.shape[0]
    full_probs = np.zeros((N, total_classes), dtype=probs.dtype)

    # model_classes contains the class indices (0..119) that the model saw
    # We map the output columns to these indices
    for i, class_idx in enumerate(model_classes):
        full_probs[:, class_idx] = probs[:, i]

    return full_probs


if __name__ == "__main__":
    print("Starting End-to-End Pipeline Demonstration...")

    # 1. Configuration Setup
    set_seed(Config.SEED)

    # Optimize for speed: Reduce CV folds and max iterations
    Config.LOGREG_PARAMS["cv"] = 2
    Config.LOGREG_PARAMS["max_iter"] = 100

    # Define subset size for demonstration (small enough for speed, large enough for logic)
    SUBSET_SIZE = 500

    # 2. Feature Extraction
    print(f"\n--- Feature Extraction (Subset: {SUBSET_SIZE}) ---")

    # Stream A: ConvNeXt
    print("Processing Stream A (ConvNeXt)...")
    train_emb_a, train_lbl_a, train_ids_a = extract_features(
        Config.STREAM_A, "train", load_cached_data=False, debug_subset_size=SUBSET_SIZE
    )
    val_emb_a, val_lbl_a, val_ids_a = extract_features(
        Config.STREAM_A, "val", load_cached_data=False, debug_subset_size=SUBSET_SIZE
    )
    test_emb_a, _, test_ids_a = extract_features(
        Config.STREAM_A, "test", load_cached_data=False, debug_subset_size=SUBSET_SIZE
    )

    # Stream B: ViT
    print("Processing Stream B (ViT)...")
    train_emb_b, train_lbl_b, train_ids_b = extract_features(
        Config.STREAM_B, "train", load_cached_data=False, debug_subset_size=SUBSET_SIZE
    )
    val_emb_b, val_lbl_b, val_ids_b = extract_features(
        Config.STREAM_B, "val", load_cached_data=False, debug_subset_size=SUBSET_SIZE
    )
    test_emb_b, _, test_ids_b = extract_features(
        Config.STREAM_B, "test", load_cached_data=False, debug_subset_size=SUBSET_SIZE
    )

    # Verify shapes
    assert train_emb_a.shape[0] == SUBSET_SIZE
    assert train_emb_b.shape[0] == SUBSET_SIZE
    assert train_emb_a.shape[1] == Config.STREAM_A["embedding_dim"] * 3  # 3 views

    # 3. Data Preparation (Filtering for valid CV)
    print("\n--- Data Preparation ---")
    # Identify classes with enough samples for CV=2
    label_counts = Counter(train_lbl_a)
    valid_classes = {cls for cls, count in label_counts.items() if count >= 2}

    print(f"Total classes in subset: {len(label_counts)}")
    print(f"Classes with >= 2 samples: {len(valid_classes)}")

    # Create mask for training data
    train_mask = np.array([lbl in valid_classes for lbl in train_lbl_a])

    # Filter Training Data
    X_train_a = train_emb_a[train_mask]
    X_train_b = train_emb_b[train_mask]
    y_train = train_lbl_a[train_mask]  # Labels are same for both streams

    print(f"Training samples after filtering: {len(y_train)}")

    # 4. Model Training
    print("\n--- Model Training ---")

    # Train Stream A
    model_a = train_stream_classifier(
        X_train_a, y_train, "demo_stream_a", load_cached_model=False
    )

    # Train Stream B
    model_b = train_stream_classifier(
        X_train_b, y_train, "demo_stream_b", load_cached_model=False
    )

    # 5. Prediction & Evaluation
    print("\n--- Prediction & Evaluation ---")

    # Predict on Validation Set
    probs_val_a_raw = predict_stream(model_a, val_emb_a)
    probs_val_b_raw = predict_stream(model_b, val_emb_b)

    # Pad probabilities to full 120 classes
    probs_val_a = pad_probabilities(probs_val_a_raw, model_a.classes_)
    probs_val_b = pad_probabilities(probs_val_b_raw, model_b.classes_)

    # Filter validation set to only evaluate on classes the model knows
    # (Otherwise log loss is undefined/infinite for unseen classes)
    val_mask = np.array([lbl in valid_classes for lbl in val_lbl_a])

    if np.sum(val_mask) > 0:
        val_lbl_eval = val_lbl_a[val_mask]
        probs_val_a_eval = probs_val_a[val_mask]
        probs_val_b_eval = probs_val_b[val_mask]

        print("Stream A Results:")
        evaluate_model(probs_val_a_eval, val_lbl_eval)

        print("Stream B Results:")
        evaluate_model(probs_val_b_eval, val_lbl_eval)

        # 6. Ensemble Optimization
        print("\n--- Ensemble Optimization ---")
        best_weight_a = optimize_ensemble_weights(
            probs_val_a_eval, probs_val_b_eval, val_lbl_eval
        )
    else:
        print(
            "Warning: No validation samples found for the trained classes. Defaulting weight to 0.5."
        )
        best_weight_a = 0.5

    # 7. Test Prediction & Submission
    print("\n--- Generating Submission ---")

    # Predict on Test Set
    probs_test_a_raw = predict_stream(model_a, test_emb_a)
    probs_test_b_raw = predict_stream(model_b, test_emb_b)

    # Pad Test Probabilities
    probs_test_a = pad_probabilities(probs_test_a_raw, model_a.classes_)
    probs_test_b = pad_probabilities(probs_test_b_raw, model_b.classes_)

    # Apply Ensemble
    final_test_probs = apply_ensemble(probs_test_a, probs_test_b, best_weight_a)

    # Create Submission
    sub_df = create_submission(test_ids_a, final_test_probs)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH)
    assert sub_df.shape == (SUBSET_SIZE, 121)  # ID + 120 breeds

    print("\nPipeline completed successfully.")
