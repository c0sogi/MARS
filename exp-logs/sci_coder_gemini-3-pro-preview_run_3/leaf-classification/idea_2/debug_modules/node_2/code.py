import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_loader import load_tabular_data
from library.feature_extractor import get_raw_image_features, reduce_dimensions
from library.model_pipeline import HybridEnsemble


def main():
    # 1. Setup and Configuration
    # Enable DEBUG mode to run on a small subset (50 samples) for speed
    Config.DEBUG = True
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("=== Configuration ===")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Cache Directory: {Config.CACHE_DIR}")
    print("-" * 30)

    # 2. Data Loading (Tabular)
    print("\n=== Loading Tabular Data ===")
    # When Config.DEBUG is True, we must manually limit the tabular loader
    # to match the behavior of the image loader which checks Config.DEBUG internally.
    limit = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None

    X_train_tab, y_train, ids_train_tab = load_tabular_data("train", limit=limit)
    X_val_tab, y_val, ids_val_tab = load_tabular_data("val", limit=limit)
    X_test_tab, _, ids_test_tab = load_tabular_data("test", limit=limit)

    print(f"Train Tabular Shape: {X_train_tab.shape}")
    print(f"Val Tabular Shape:   {X_val_tab.shape}")
    print(f"Test Tabular Shape:  {X_test_tab.shape}")

    # Verify tabular feature count (margin + shape + texture = 64*3 = 192)
    assert X_train_tab.shape[1] == 192, "Tabular data should have 192 features"

    # 3. Feature Extraction (Image)
    print("\n=== Extracting Image Features ===")
    # get_raw_image_features checks Config.DEBUG internally and slices the metadata
    X_train_img_raw, ids_train_img = get_raw_image_features("train")
    X_val_img_raw, ids_val_img = get_raw_image_features("val")
    X_test_img_raw, ids_test_img = get_raw_image_features("test")

    print(f"Train Image Raw Shape: {X_train_img_raw.shape}")

    # 4. Data Alignment Verification
    # Ensure that the tabular data and image data correspond to the same samples in the same order
    print("\n=== Verifying Data Alignment ===")
    assert np.array_equal(
        ids_train_tab, ids_train_img
    ), "Train IDs mismatch between tabular and image data"
    assert np.array_equal(
        ids_val_tab, ids_val_img
    ), "Val IDs mismatch between tabular and image data"
    assert np.array_equal(
        ids_test_tab, ids_test_img
    ), "Test IDs mismatch between tabular and image data"
    print("Data alignment confirmed.")

    # 5. Dimensionality Reduction (PCA)
    print("\n=== Applying PCA to Image Features ===")
    # Reduce the 2048-dim ResNet features to retain 95% variance
    X_train_img_pca, X_val_img_pca, X_test_img_pca = reduce_dimensions(
        X_train_img_raw, X_val_img_raw, X_test_img_raw
    )

    print(f"PCA Components retained: {X_train_img_pca.shape[1]}")
    print(f"Train Image PCA Shape: {X_train_img_pca.shape}")

    # 6. Model Training
    print("\n=== Training Hybrid Ensemble ===")
    ensemble = HybridEnsemble()

    # Train the ensemble (Logistic Regression + LDA)
    # This includes scaling, feature fusion, and hyperparameter tuning on the validation set
    ensemble.train_models(
        X_train_tab, X_train_img_pca, y_train, X_val_tab, X_val_img_pca, y_val
    )

    # 7. Prediction
    print("\n=== Generating Predictions ===")
    preds, class_names = ensemble.predict_ensemble(X_test_tab, X_test_img_pca)

    # Handle Disjoint Label Sets When Subsampling High-Cardinality Data (Cite debug_lesson_1)
    # In DEBUG mode, the model might not see all 99 classes. We align predictions to the full schema.
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    all_classes = [c for c in sample_sub.columns if c != "id"]

    if preds.shape[1] != len(all_classes):
        print(
            f"Aligning predictions from {preds.shape[1]} classes to {len(all_classes)} classes..."
        )
        pred_df = pd.DataFrame(preds, columns=class_names)
        # Reindex to include all classes, filling missing ones with 0.0
        pred_df = pred_df.reindex(columns=all_classes, fill_value=0.0)
        preds = pred_df.values
        class_names = np.array(all_classes)

    print(f"Prediction Shape: {preds.shape}")

    # Verify prediction shape: (n_samples, n_classes)
    # In debug mode, n_samples is min(DEBUG_SAMPLE_SIZE, actual_test_size)
    expected_rows = min(limit, 99) if limit is not None else 99
    assert (
        preds.shape[0] == expected_rows
    ), f"Expected {expected_rows} predictions, got {preds.shape[0]}"
    assert preds.shape[1] == 99, f"Expected 99 classes, got {preds.shape[1]}"

    # 8. Submission
    print("\n=== Saving Submission ===")
    save_submission(preds, ids_test_tab, class_names, Config.SUBMISSION_FILE_PATH)

    # Verify file creation
    if os.path.exists(Config.SUBMISSION_FILE_PATH):
        print(f"Successfully created submission file at {Config.SUBMISSION_FILE_PATH}")

        # Quick check of file content
        df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
        print(f"Submission File Rows: {len(df_sub)}")
        assert len(df_sub) == expected_rows, "Submission file row count mismatch"
        assert "id" in df_sub.columns, "Submission file missing 'id' column"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Pipeline Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
