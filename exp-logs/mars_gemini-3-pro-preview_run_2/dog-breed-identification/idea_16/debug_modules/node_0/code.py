import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

# Import provided library modules
import library.config as config
import library.data_loader as dl
import library.model_utils as mu
import library.feature_processing as fp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # 1. Setup and Configuration
    config.seed_everything()
    device = config.DEVICE
    print(f"Device: {device}")

    # 2. Load Metadata and Create Subsets (Optimization for Speed)
    print("\n--- Step 1: Loading and Subsampling Data ---")
    train_df = pd.read_csv(config.TRAIN_METADATA)
    val_df = pd.read_csv(config.VAL_METADATA)
    test_df = pd.read_csv(config.TEST_METADATA)

    # Subsample to ensure quick execution
    # We select a subset of classes to ensure stratification isn't an issue for this tiny demo,
    # but for simplicity, we just take the head.
    N_TRAIN = 50
    N_VAL = 20
    N_TEST = 20

    train_subset = train_df.head(N_TRAIN).copy()
    val_subset = val_df.head(N_VAL).copy()
    test_subset = test_df.head(N_TEST).copy()

    print(f"Train subset: {len(train_subset)}")
    print(f"Val subset:   {len(val_subset)}")
    print(f"Test subset:  {len(test_subset)}")

    # 3. Feature Extraction Loop
    print("\n--- Step 2: Running Inference (Feature Extraction) ---")

    # Get transforms for all views
    view_transforms = dl.get_view_transforms()

    # Storage for raw features: {view: (s3, s4, ids, labels)}
    raw_train = {}
    raw_val = {}
    raw_test = {}

    # We use custom dataset names to avoid conflict with full-dataset caches if they existed
    # and to force the logic to run (though we also pass load_cached_data=False)
    datasets = [
        ("demo_train", train_subset, raw_train),
        ("demo_val", val_subset, raw_val),
        ("demo_test", test_subset, raw_test),
    ]

    for view_name in config.VIEWS.keys():
        print(f"Processing View: {view_name}")
        transform = view_transforms[view_name]

        for ds_name, df, storage_dict in datasets:
            # Instantiate Dataset
            dataset = dl.DogDataset(df, transform=transform)

            # Instantiate DataLoader
            # Using a smaller batch size for the demo if needed, but config.BATCH_SIZE is fine
            loader = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=2,  # Reduced workers for small subset
            )

            # Run Inference
            # load_cached_data=False forces the extraction logic to run
            s3, s4, ids, labels = mu.run_inference(
                loader,
                dataset_name=ds_name,
                view_name=view_name,
                device=device,
                load_cached_data=False,
            )

            # Verify Shapes
            # ConvNeXt Large: Stage 3 (Texture) usually 768 or 1536 depending on definition.
            # Based on torchvision implementation:
            # features[5] (Stage 2 in 0-idx) -> 768 channels
            # features[7] (Stage 3 in 0-idx) -> 1536 channels (Final)
            # Let's verify this assumption with assertions.

            # Expected: (N_SAMPLES, CHANNELS)
            assert s3.shape[0] == len(
                df
            ), f"S3 batch size mismatch: {s3.shape[0]} vs {len(df)}"
            assert s4.shape[0] == len(
                df
            ), f"S4 batch size mismatch: {s4.shape[0]} vs {len(df)}"

            # Store results
            storage_dict[view_name] = (s3, s4, ids, labels)

    print("Feature extraction complete.")

    # Check feature dimensions from one sample
    sample_s3 = raw_train["global"][0]
    sample_s4 = raw_train["global"][1]
    print(
        f"Feature Dimensions -> Stage 3: {sample_s3.shape[1]}, Stage 4: {sample_s4.shape[1]}"
    )

    # 4. Feature Processing (Fusion)
    print("\n--- Step 3: Feature Processing (Normalization & Fusion) ---")

    # This will normalize Stage 3 features and concatenate everything
    (
        (train_X, train_ids, train_y),
        (val_X, val_ids, val_y),
        (test_X, test_ids, test_y),
    ) = fp.process_features(raw_train, raw_val, raw_test, load_cached_data=False)

    print(f"Fused Train Shape: {train_X.shape}")
    print(f"Fused Val Shape:   {val_X.shape}")
    print(f"Fused Test Shape:  {test_X.shape}")

    # Verify Fused Dimensions
    # 3 Views * (Stage3_Dim + Stage4_Dim)
    expected_dim = 3 * (sample_s3.shape[1] + sample_s4.shape[1])
    assert (
        train_X.shape[1] == expected_dim
    ), f"Fused dimension mismatch: {train_X.shape[1]} != {expected_dim}"

    # 5. Model Training (Logistic Regression)
    print("\n--- Step 4: Training Classifier ---")

    # Initialize Classifier
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=100,
        random_state=config.SEED,
        verbose=0,
    )

    # Fit
    clf.fit(train_X, train_y)
    print("Model trained.")

    # 6. Validation
    print("\n--- Step 5: Validation ---")
    val_probs = clf.predict_proba(val_X)

    # Calculate Log Loss
    # Note: In this tiny subset, we might not have all classes in the training set.
    # Scikit-learn handles this by only outputting probs for seen classes.
    # For a robust metric calculation in this demo, we check the labels.

    # Get the classes the model learned
    learned_classes = clf.classes_
    print(f"Model learned {len(learned_classes)} classes from the subset.")

    # Filter validation set to only include classes seen in training (for demo purposes only)
    # In a real run, the training set covers all classes.
    mask = np.isin(val_y, learned_classes)
    if mask.sum() > 0:
        filtered_val_y = val_y[mask]
        filtered_val_probs = val_probs[mask]
        score = log_loss(filtered_val_y, filtered_val_probs, labels=learned_classes)
        print(f"Validation Log Loss (Subset): {score:.4f}")
    else:
        print(
            "Validation subset contained no classes seen in training subset (expected for tiny random subsets)."
        )

    # 7. Generate Submission
    print("\n--- Step 6: Generating Submission ---")
    test_probs = clf.predict_proba(test_X)

    # We need to map these probabilities to the full 120 classes required by the submission format.
    # 1. Get all unique breeds from the full training metadata to ensure correct column order.
    all_breeds = sorted(pd.read_csv(config.TRAIN_METADATA)["breed"].unique())
    assert len(all_breeds) == 120, "Expected 120 breeds."

    # 2. Create a placeholder array for submission (N_TEST, 120)
    submission_probs = np.zeros((len(test_subset), len(all_breeds)))

    # 3. Map learned class probabilities to the correct columns
    # clf.classes_ contains the string labels of breeds seen during training
    class_to_idx = {b: i for i, b in enumerate(all_breeds)}

    for i, learned_class in enumerate(learned_classes):
        if learned_class in class_to_idx:
            col_idx = class_to_idx[learned_class]
            submission_probs[:, col_idx] = test_probs[:, i]

    # 4. Create DataFrame
    submission_df = pd.DataFrame(submission_probs, columns=all_breeds)
    submission_df.insert(0, "id", test_ids)

    # 5. Save
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Verify submission format
    assert submission_df.shape == (
        N_TEST,
        121,
    ), f"Submission shape mismatch: {submission_df.shape}"
    print("Submission format verified.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
