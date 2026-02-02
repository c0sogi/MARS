import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, accuracy_score

# Import from the provided library
from library.config import set_seed, TRAIN_META_PATH, INPUT_DIR, DEVICE, FLOAT_PRECISION
from library.feature_extraction import extract_single_image_features
from library.data_pipeline import get_data_pipeline
from library.oas_model import OASDiscriminant, train_oas_model
from library.gpu_inference import predict_proba_gpu


def run_demo():
    # 1. Setup and Reproducibility
    print("1. Setting up environment...")
    set_seed(42)

    # 2. Demonstrate Feature Extraction
    # We will pick one image from the training metadata to test the extractor
    print("\n2. Demonstrating Feature Extraction...")
    if os.path.exists(TRAIN_META_PATH):
        df_train = pd.read_csv(TRAIN_META_PATH)
        # Get the first image path
        sample_rel_path = df_train.iloc[0]["file_path"]
        print(f"   Extracting features for: {sample_rel_path}")

        # Extract features
        features = extract_single_image_features(sample_rel_path)

        # Validation
        expected_keys = [
            "geo_Area",
            "geo_Major_Axis_Length",
            "geo_Mean_Thickness",
            "geo_Eccentricity",
            "geo_Solidity",
            "geo_Extent",
            "geo_Aspect_Ratio",
        ]
        assert isinstance(features, dict), "Output must be a dictionary"
        assert all(k in features for k in expected_keys), "Missing geometric features"
        assert all(
            isinstance(v, float) for v in features.values()
        ), "Features must be floats"
        print(
            "   Feature extraction successful. Sample features:", list(features.keys())
        )
    else:
        print(
            "   Warning: Metadata file not found, skipping specific image extraction test."
        )

    # 3. Demonstrate Data Pipeline
    # We use a debug subset size to ensure this runs quickly within the time limit
    print("\n3. Running Data Pipeline (Subset)...")
    subset_size = 100
    data_dict = get_data_pipeline(load_cached_data=True, debug_subset_size=subset_size)

    # Unpack data
    X_train, y_train = data_dict["train"]
    X_val, y_val = data_dict["val"]
    X_test, test_ids = data_dict["test"]
    le = data_dict["label_encoder"]
    feature_names = data_dict["feature_names"]

    # Validation
    print(f"   Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    assert X_train.shape[0] == subset_size, f"Expected {subset_size} training samples"
    # Cite debug_lesson_6: Dynamic Subsampling Requires Dynamic Assertions
    # Validation size may be smaller due to class filtering in debug mode
    assert (
        X_val.shape[0] <= subset_size
    ), f"Expected <= {subset_size} validation samples"
    assert X_train.shape[1] == len(feature_names), "Feature count mismatch"
    assert X_train.dtype == np.float64, "Pipeline must return float64 data"

    # 4. Demonstrate OAS Model Training
    print("\n4. Training OAS Discriminant Model...")
    model = OASDiscriminant()
    model.fit(X_train, y_train)

    # Check model internal state
    W, b, classes = model.get_linear_parameters()
    print(f"   Model fitted. Weights shape: {W.shape}, Bias shape: {b.shape}")
    assert W.shape[0] == len(classes), "Weights class dim mismatch"
    assert W.shape[1] == X_train.shape[1], "Weights feature dim mismatch"

    # 5. Inference and Evaluation
    print("\n5. Performing Inference and Evaluation...")
    # Predict on validation set
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)

    # Calculate metrics
    loss = log_loss(y_val, val_probs)
    acc = accuracy_score(y_val, val_preds)

    print(f"   Validation Log Loss: {loss:.4f}")
    print(f"   Validation Accuracy: {acc:.4f}")

    # Validate probabilities
    assert val_probs.shape == (
        X_val.shape[0],
        len(classes),
    ), "Probability shape mismatch"
    assert np.allclose(val_probs.sum(axis=1), 1.0), "Probabilities must sum to 1"
    assert (
        val_probs.min() >= 0 and val_probs.max() <= 1
    ), "Probabilities must be in [0, 1]"

    # 6. Demonstrate GPU Inference Kernel
    print("\n6. Verifying GPU Inference Kernel...")
    if torch.cuda.is_available():
        print(f"   Testing on {DEVICE}...")
        # Run the standalone GPU inference function
        gpu_probs = predict_proba_gpu(X_val, W, b)

        # Compare with the class method (which also uses GPU internally)
        # They should be identical as they use the same logic/precision
        diff = np.abs(val_probs - gpu_probs).max()
        print(
            f"   Max difference between Class Inference and Standalone Kernel: {diff:.16f}"
        )
        assert diff < 1e-14, "GPU Inference kernel results diverge from model method"
    else:
        print(
            "   CUDA not available. Skipping explicit GPU kernel verification (CPU fallback used)."
        )

    # 7. Generate Submission
    print("\n7. Generating Demo Submission...")
    # Predict on test set
    test_probs = model.predict_proba(X_test)

    # Create DataFrame
    # Columns must be the class names decoded from the label encoder
    class_names = le.inverse_transform(classes)
    submission_df = pd.DataFrame(test_probs, columns=class_names)
    submission_df.insert(0, "id", test_ids)

    # Save to working directory
    output_path = "./working/demo_submission.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"   Submission saved to {output_path}")
    print(f"   Submission shape: {submission_df.shape}")

    # Validate submission format
    assert "id" in submission_df.columns
    assert len(submission_df) == subset_size
    assert submission_df.shape[1] == len(class_names) + 1

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
