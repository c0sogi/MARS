import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_macro_f1
from library.dataset import create_dataloaders
from library.feature_extractor import FeatureModel, get_features
from library.model import LinearProbe, train_model, generate_submission
from library.trainer import run_training


def main():
    print("==== Starting Demonstration Script ====")

    # -------------------------------------------------------------------------
    # 1. Test Utilities
    # -------------------------------------------------------------------------
    print("\n[Demo] Testing Utilities...")
    set_seed(42)

    # verify metric calculation
    y_true_mock = [0, 1, 2, 0, 1, 2]
    y_pred_mock = [0, 1, 2, 0, 1, 2]
    f1 = calculate_macro_f1(y_true_mock, y_pred_mock)
    assert f1 == 1.0, f"Metric calculation failed. Expected 1.0, got {f1}"
    print("Utils verified: Seed set and F1 score calculated correctly.")

    # -------------------------------------------------------------------------
    # 2. Test Dataset and DataLoaders
    # -------------------------------------------------------------------------
    print("\n[Demo] Testing Dataset and DataLoaders...")
    # Use a small sample size to keep execution fast
    SAMPLE_SIZE = 32
    BATCH_SIZE = 8

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, sample_size=SAMPLE_SIZE
    )

    # Verify Train Loader structure
    for images, targets in train_loader:
        # Expected: (Batch, 3, 224, 224) and (Batch,)
        assert images.shape == (
            BATCH_SIZE,
            3,
            224,
            224,
        ), f"Train Image shape mismatch: {images.shape}"
        assert targets.shape == (
            BATCH_SIZE,
        ), f"Train Target shape mismatch: {targets.shape}"
        assert targets.dtype == torch.long, "Targets must be LongTensor"
        break  # Check only first batch

    # Verify Test Loader structure (returns images, ids)
    for images, ids in test_loader:
        assert images.shape == (
            BATCH_SIZE,
            3,
            224,
            224,
        ), f"Test Image shape mismatch: {images.shape}"
        assert len(ids) == BATCH_SIZE, "Test IDs batch size mismatch"
        break

    print(f"DataLoaders verified with sample_size={SAMPLE_SIZE}.")

    # -------------------------------------------------------------------------
    # 3. Test Feature Extractor
    # -------------------------------------------------------------------------
    print("\n[Demo] Testing Feature Extractor...")

    # A. Verify Model Architecture
    model = FeatureModel()
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    # ResNet50 (2048) -> AvgPool (2048) + MaxPool (2048) -> Concat (4096)
    assert output.shape == (2, 4096), f"Feature output shape mismatch: {output.shape}"
    print("FeatureModel architecture verified (Output dim: 4096).")

    # B. Verify Extraction Pipeline
    # We set load_cached_data=False to force the code to run the extraction loop
    print("Running feature extraction on small subset (forcing re-computation)...")
    train_feats, train_targets = get_features(
        train_loader, mode="train", load_cached_data=False
    )
    val_feats, val_targets = get_features(
        val_loader, mode="val", load_cached_data=False
    )
    test_feats, test_ids = get_features(
        test_loader, mode="test", load_cached_data=False
    )

    # Assertions
    assert train_feats.shape[1] == 4096, "Train feature dimension incorrect"
    assert len(train_feats) == len(
        train_targets
    ), "Train features/targets length mismatch"
    assert len(test_feats) == len(test_ids), "Test features/ids length mismatch"

    # Verify Cache Creation
    assert os.path.exists(
        Config.TRAIN_FEATURES
    ), "Train features cache file not created"
    assert os.path.exists(Config.TEST_IDS), "Test IDs cache file not created"

    print("Feature extraction pipeline and caching verified.")

    # -------------------------------------------------------------------------
    # 4. Test Model Training
    # -------------------------------------------------------------------------
    print("\n[Demo] Testing Model Training...")

    # A. Verify Linear Probe
    probe = LinearProbe(input_dim=4096, num_classes=Config.NUM_CLASSES)
    dummy_feats = torch.randn(4, 4096)
    logits = probe(dummy_feats)
    assert logits.shape == (
        4,
        Config.NUM_CLASSES,
    ), f"Logits shape mismatch: {logits.shape}"

    # B. Run Training Loop
    # Using the extracted features from step 3
    print("Training Linear Probe (2 Epochs)...")
    trained_model = train_model(
        train_feats, train_targets, val_feats, val_targets, epochs=2, lr=0.1, patience=1
    )

    assert isinstance(
        trained_model, LinearProbe
    ), "train_model did not return a LinearProbe instance"

    # Check if model weights are on the correct device
    param = next(trained_model.parameters())
    expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert expected_device_type in str(
        param.device
    ), f"Model device mismatch. Expected {expected_device_type}, got {param.device}"

    print("Model training verified.")

    # -------------------------------------------------------------------------
    # 5. Test Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Demo] Testing Submission Generation...")

    output_csv_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    generate_submission(
        trained_model, test_feats, test_ids, output_path=output_csv_path
    )

    assert os.path.exists(output_csv_path), "Submission file was not created"

    # Verify content format
    df_sub = pd.read_csv(output_csv_path)
    assert list(df_sub.columns) == [
        "Id",
        "Predicted",
    ], f"Submission columns incorrect. Found: {list(df_sub.columns)}"
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count mismatch. Expected {len(test_ids)}, got {len(df_sub)}"
    assert df_sub["Predicted"].dtype == "int64", "Predicted column should be integer"

    print(f"Submission generation verified. File saved to {output_csv_path}")

    # -------------------------------------------------------------------------
    # 6. Integration Test (Full Pipeline)
    # -------------------------------------------------------------------------
    print("\n[Demo] Testing Full Trainer Pipeline (Integration Test)...")

    # Run the orchestrator function with minimal settings
    # This ensures all components talk to each other correctly
    run_training(
        load_cached_data=False,  # Force overwrite of cache with new sample size
        sample_size=16,
        epochs=1,
        lr=0.1,
    )

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Final submission from trainer not found"
    print("Full trainer pipeline execution successful.")

    print("\n==== All Demonstrations Completed Successfully ====")


if __name__ == "__main__":
    main()
