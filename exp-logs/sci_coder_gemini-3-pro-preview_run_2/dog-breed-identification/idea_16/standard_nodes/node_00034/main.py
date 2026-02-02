import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
import library.config as config
import library.data_loader as data_loader
import library.model_utils as model_utils
import library.feature_processing as feature_processing


def main():
    # 1. Initialization
    config.seed_everything()
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(config.TRAIN_METADATA)
    val_df = pd.read_csv(config.VAL_METADATA)
    test_df = pd.read_csv(config.TEST_METADATA)

    # 3. Feature Extraction Loop
    # We need to collect raw features for all views: global, standard, local
    views = ["global", "standard", "local"]

    # Dictionaries to hold raw feature tuples: (s3, s4, ids, labels)
    raw_train = {}
    raw_val = {}
    raw_test = {}

    # Get transforms for all views
    view_transforms = data_loader.get_view_transforms()

    for view in views:
        print(f"\n--- Processing View: {view} ---")
        transform = view_transforms[view]

        # Create Datasets
        train_ds = data_loader.DogDataset(train_df, transform=transform)
        val_ds = data_loader.DogDataset(val_df, transform=transform)
        test_ds = data_loader.DogDataset(test_df, transform=transform)

        # Create DataLoaders
        # Using num_workers from config, pin_memory for GPU speed
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Inference (Extract Features)
        # load_cached_data=True allows skipping inference if files exist in ./working
        raw_train[view] = model_utils.run_inference(
            train_loader, "train", view, device, load_cached_data=True
        )
        raw_val[view] = model_utils.run_inference(
            val_loader, "val", view, device, load_cached_data=True
        )
        raw_test[view] = model_utils.run_inference(
            test_loader, "test", view, device, load_cached_data=True
        )

    # 4. Feature Processing (Normalization & Fusion)
    print("\n--- Fusing Features ---")
    # This handles: Alignment, S3 Normalization (Fit on Train, Transform Val/Test), Concatenation
    (
        (train_X, train_ids, train_y),
        (val_X, val_ids, val_y),
        (test_X, test_ids, test_y),
    ) = feature_processing.process_features(
        raw_train, raw_val, raw_test, load_cached_data=True
    )

    print(f"Fused Train Shape: {train_X.shape}")
    print(f"Fused Val Shape:   {val_X.shape}")

    # 5. Model Training
    print("\n--- Training Model ---")
    # LogisticRegressionCV for automatic hyperparameter tuning
    # n_jobs=-1 uses all CPUs
    clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=500,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(train_X, train_y)
    print("Training complete.")

    # 6. Validation
    print("\n--- Validating ---")
    val_probs = clf.predict_proba(val_X)

    # Calculate Metric
    # log_loss requires (y_true, y_pred_probs, labels=classes)
    metric = log_loss(val_y, val_probs, labels=clf.classes_)

    print(f"Final Validation Metric: {metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample log loss
    # We need to index into val_probs using the integer index of the true class

    # Map string labels to indices based on clf.classes_
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    y_indices = np.array([class_to_idx[lbl] for lbl in val_y])

    # Extract probability assigned to the true class
    # Clip to avoid log(0)
    epsilon = 1e-15
    prob_true = val_probs[np.arange(len(val_y)), y_indices]
    prob_true = np.clip(prob_true, epsilon, 1 - epsilon)
    sample_losses = -np.log(prob_true)

    # Calculate Feature Signal Magnitude (L2 Norm)
    feature_norms = np.linalg.norm(val_X, axis=1)

    # Correlation
    correlation = np.corrcoef(sample_losses, feature_norms)[0, 1]
    print(
        f"Correlation between Error Magnitude and Feature Signal Norm: {correlation:.6f}"
    )

    # 8. Submission
    threshold = 0.11640673500383826
    if metric < threshold:
        print(
            f"\nMetric ({metric}) is better than threshold ({threshold}). Generating submission..."
        )

        test_probs = clf.predict_proba(test_X)

        # Create DataFrame
        # Columns must be sorted breeds. clf.classes_ is sorted alphabetically by default for strings.
        submission_df = pd.DataFrame(test_probs, columns=clf.classes_)

        # Add ID column
        submission_df.insert(0, "id", test_ids)

        # Save
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nMetric ({metric}) did not beat threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
