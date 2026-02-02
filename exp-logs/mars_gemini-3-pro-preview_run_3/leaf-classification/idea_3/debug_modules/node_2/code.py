import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_data
from library.feature_extraction import FeatureExtractor
from library.preprocessing import FusionPipeline
from library.model import EnsembleClassifier


def main():
    print("=== Leaf Classification Pipeline Demo ===")

    # 1. Setup & Configuration
    # We override specific Config parameters to ensure the demo runs quickly
    # and doesn't overwrite existing production files.
    print("\n[1] Configuration Setup")
    seed_everything(42)
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples per split for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.setup()

    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n[2] Loading Data")
    # load_data handles metadata reading and dataset creation
    train_ds, val_ds, test_ds, le = load_data(debug=True)

    print(f"    Train size: {len(train_ds)}")
    print(f"    Val size:   {len(val_ds)}")
    print(f"    Test size:  {len(test_ds)}")
    print(f"    Total Species: {len(le.classes_)}")

    # Verify dataset item structure
    # Item: (images, features, label, id)
    sample_img, sample_feat, sample_lbl, sample_id = train_ds[0]

    # Assertions to verify correctness
    assert sample_img.shape == (
        4,
        3,
        224,
        224,
    ), f"Image tensor shape incorrect: {sample_img.shape}"
    assert sample_feat.shape == (
        192,
    ), f"Tabular feature shape incorrect: {sample_feat.shape}"
    assert isinstance(sample_id, (int, np.integer)), "ID should be an integer"
    print("    Dataset item structure verified.")

    # 3. Feature Extraction
    print("\n[3] Feature Extraction (CNN + ViT + Tabular)")
    extractor = FeatureExtractor()

    # Run extraction (load_cached_data=False forces execution)
    # This uses the ResNet50 and ViT backbones defined in Config
    features_dict = extractor.run(load_cached_data=False)

    # Verify output structure
    expected_splits = ["train", "val", "test"]
    expected_keys = ["cnn", "vit", "tab"]

    for split in expected_splits:
        assert split in features_dict, f"Missing split {split} in features"
        for key in expected_keys:
            assert key in features_dict[split], f"Missing key {key} in {split}"

    # Verify shapes correspond to dataset size
    n_train = len(train_ds)
    assert features_dict["train"]["cnn"].shape[0] == n_train
    assert features_dict["train"]["vit"].shape[0] == n_train
    assert features_dict["train"]["tab"].shape[0] == n_train
    print("    Feature extraction complete. Shapes verified.")

    # 4. Preprocessing & Fusion
    print("\n[4] Preprocessing & Feature Fusion")
    pipeline = FusionPipeline()

    # Fit scalers and PCA on training data
    pipeline.fit(features_dict["train"], load_cached_data=False)

    # Transform all splits
    X_train = pipeline.transform(features_dict["train"])
    X_val = pipeline.transform(features_dict["val"])
    X_test = pipeline.transform(features_dict["test"])

    # Get targets
    y_train = features_dict["train"]["tgt"]
    y_val = features_dict["val"]["tgt"]
    test_ids = features_dict["test"]["ids"]

    # Verify fused shape
    # Shape should be N x (192 + PCA_CNN + PCA_VIT)
    # Since N=20 is small, PCA components will be limited by N
    print(f"    Fused Train Feature Shape: {X_train.shape}")
    assert len(X_train.shape) == 2
    assert X_train.shape[0] == n_train
    print("    Preprocessing pipeline verified.")

    # 5. Model Training
    print("\n[5] Model Training (Ensemble)")
    model = EnsembleClassifier()

    # Train LR and LDA
    model.fit(X_train, y_train)

    # Check which classes the model actually learned (since we used a tiny subset)
    trained_classes = model.lr.classes_
    print(
        f"    Model trained on {len(trained_classes)} unique classes found in debug subset."
    )

    # 6. Optimization
    print("\n[6] Ensemble Weight Optimization")
    # We optimize the blend weight on validation data.
    # Note: For this demo, we must filter validation data to only include classes
    # that were present in the training subset, otherwise log_loss calculation fails.

    val_mask = np.isin(y_val, trained_classes)
    if val_mask.sum() > 0:
        X_val_filtered = X_val[val_mask]
        y_val_filtered = y_val[val_mask]

        best_weight = model.optimize_ensemble_weight(X_val_filtered, y_val_filtered)
        print(f"    Optimization successful. Best LDA weight: {best_weight}")
    else:
        print(
            "    Skipping optimization (No overlapping classes in debug validation subset)."
        )
        best_weight = 0.5

    # 7. Prediction & Submission Generation
    print("\n[7] Generating Submission")
    # Predict probabilities on test set
    # Returns shape (N_test, n_trained_classes)
    raw_probs = model.predict(X_test, weight=best_weight)

    # The submission requires columns for ALL 99 classes.
    # We need to map the model's output columns to the correct full-dataset columns.

    # Initialize full probability matrix with epsilon
    n_test_samples = len(test_ids)
    n_total_classes = len(le.classes_)
    full_probs = np.ones((n_test_samples, n_total_classes)) * 1e-15

    # Map predictions
    # trained_classes contains the global indices of the classes the model knows
    for i, global_class_idx in enumerate(trained_classes):
        full_probs[:, global_class_idx] = raw_probs[:, i]

    # Renormalize rows to sum to 1
    row_sums = full_probs.sum(axis=1, keepdims=True)
    full_probs = full_probs / row_sums

    # Construct DataFrame
    submission_df = pd.DataFrame(full_probs, columns=le.classes_)
    submission_df.insert(0, "id", test_ids)

    # Verify Schema
    print("    Submission Head:")
    print(submission_df.head(2).iloc[:, :5])  # Print first 5 cols

    assert submission_df.shape == (
        n_test_samples,
        100,
    ), "Submission shape mismatch (should be N x 100)"
    assert "id" in submission_df.columns
    assert not submission_df.isnull().values.any(), "Submission contains NaNs"

    # Save
    output_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"    Submission saved to: {output_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
