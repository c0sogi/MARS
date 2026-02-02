import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config, feature_extractor, classifier, data_utils


def main():
    # 1. Setup
    config.set_seed()
    print("Initializing Robust Multi-View Early Fusion Pipeline...")

    # 2. Feature Extraction
    # We extract features for three distinct views to capture Shape (Global), Context (Standard), and Texture (Local)
    views = ["global", "standard", "local"]
    splits = ["train", "val", "test"]

    # Dictionaries to store extracted data
    # Structure: data[split][view] -> features
    extracted_features = {split: [] for split in splits}
    labels = {}
    ids = {}

    for split in splits:
        for view in views:
            print(f"--- Processing Split: {split} | View: {view} ---")
            # Extract features using the library function which handles caching and multi-crop aggregation
            feats, lbls, img_ids = feature_extractor.get_features(
                split=split,
                view_type=view,
                load_cached_data=True,
                debug_sample_size=None,  # Use full dataset for best score
            )

            extracted_features[split].append(feats)

            # Store labels and IDs from the first view (they are consistent across views)
            if view == views[0]:
                labels[split] = lbls
                ids[split] = img_ids

    # 3. Early Fusion
    print("\n--- Performing Early Fusion ---")
    # Concatenate features from all views along the feature dimension (axis 1)
    X_train = np.concatenate(extracted_features["train"], axis=1)
    y_train = labels["train"]

    X_val = np.concatenate(extracted_features["val"], axis=1)
    y_val = labels["val"]

    X_test = np.concatenate(extracted_features["test"], axis=1)
    test_ids = ids["test"]

    print(f"Fused Training Data Shape: {X_train.shape}")
    print(f"Fused Validation Data Shape: {X_val.shape}")
    print(f"Fused Test Data Shape: {X_test.shape}")

    # 4. Model Training
    print("\n--- Training Classifier ---")
    # Train LogisticRegressionCV on the fused features
    model = classifier.train_classifier(X_train, y_train, load_cached_data=True)

    # 5. Validation & Evaluation
    print("\n--- Validating Model ---")
    # Calculate Multi Class Log Loss on the hold-out validation set
    val_loss = classifier.evaluate_model(model, X_val, y_val)

    # REQUIRED: Print the final validation metric in the exact format
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("\n--- Performing Failure Analysis ---")
    # Calculate per-sample log loss
    val_probs = model.predict_proba(X_val)

    # Clip probabilities for numerical stability in manual log loss calculation
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)

    # Extract the probability assigned to the true class for each sample
    # y_val contains class indices
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val.astype(int)]
    sample_losses = -np.log(true_class_probs)

    # Load metadata to analyze correlations
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    val_df["loss"] = sample_losses

    # Analysis 1: Correlation with Class Frequency (Training Data)
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    breed_counts = train_df["breed"].value_counts()
    val_df["train_freq"] = val_df["breed"].map(breed_counts)

    freq_corr = val_df["loss"].corr(val_df["train_freq"])
    print(f"Correlation between Error (Log Loss) and Class Frequency: {freq_corr:.4f}")

    # Analysis 2: Correlation with Feature Magnitude (Signal Strength)
    # Calculate L2 norm of the fused feature vector for each validation sample
    feature_norms = np.linalg.norm(X_val, axis=1)
    val_df["feature_norm"] = feature_norms

    norm_corr = val_df["loss"].corr(val_df["feature_norm"])
    print(
        f"Correlation between Error (Log Loss) and Feature Vector Norm: {norm_corr:.4f}"
    )

    # 7. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.11640673500383826

    if val_loss < THRESHOLD:
        print(
            f"\nValidation loss ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate probabilities for test set
        test_probs = model.predict_proba(X_test)

        # Get class names in correct order (alphabetical/index-based)
        _, idx_to_class = data_utils.get_class_mapping()
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        # Create submission file
        classifier.create_submission(test_ids, test_probs, class_names)
    else:
        print(
            f"\nValidation loss ({val_loss}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
