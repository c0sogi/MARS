import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import TabularFeatureExtractor
from library.spectrogram_processing import SpectrogramGenerator
from library.data_loaders import SeismicCNNDataset, get_spectrogram_loaders
from library.training_engines import CNNTrainer, LGBMTrainer, RidgeStacker
from library.cross_validation import CrossValidator

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_metadata():
    """
    Creates mini metadata files in the working directory referencing
    actual files present in the input directory.
    """
    print("Creating mini metadata files...")

    # Define valid segment IDs based on the dataset description provided
    # Train files: 1000015382, 1000554676, 1000745424, 1001461087
    # Test files: 1003520023, 1004346803, 1007996426, 1009749143

    train_data = [
        {
            "segment_id": 1000015382,
            "time_to_eruption": 10000,
            "file_path": "train/1000015382.csv",
        },
        {
            "segment_id": 1000554676,
            "time_to_eruption": 20000,
            "file_path": "train/1000554676.csv",
        },
        {
            "segment_id": 1000745424,
            "time_to_eruption": 30000,
            "file_path": "train/1000745424.csv",
        },
        {
            "segment_id": 1001461087,
            "time_to_eruption": 40000,
            "file_path": "train/1001461087.csv",
        },
    ]

    # Split into mini train and val
    df_train = pd.DataFrame(train_data[:2])
    df_val = pd.DataFrame(train_data[2:])

    test_data = [
        {
            "segment_id": 1003520023,
            "time_to_eruption": 0,
            "file_path": "test/1003520023.csv",
        },
        {
            "segment_id": 1004346803,
            "time_to_eruption": 0,
            "file_path": "test/1004346803.csv",
        },
    ]
    df_test = pd.DataFrame(test_data)

    # Save to working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    df_train.to_csv(mini_train_path, index=False)
    df_val.to_csv(mini_val_path, index=False)
    df_test.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def configure_environment(mini_train_path, mini_val_path, mini_test_path):
    """
    Monkey-patches the Config class to use mini datasets and faster training parameters.
    """
    print("Configuring environment for fast execution...")

    # Override Metadata Paths
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    # Override Training Parameters for Speed
    Config.N_FOLDS = 2
    Config.CNN_EPOCHS = 1
    Config.CNN_BATCH_SIZE = 2
    Config.LGB_PARAMS["n_estimators"] = 10
    Config.LGB_PARAMS["verbosity"] = -1

    # Ensure working directory is clean/ready
    if os.path.exists(Config.WORKING_DIR):
        # Don't delete the directory itself as we just put metadata there,
        # but we can clear cache files if needed. For now, we keep it simple.
        pass
    else:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)


def demo_tabular_features():
    print("\n=== Demo: Tabular Feature Extraction (Branch A) ===")
    extractor = TabularFeatureExtractor()

    # Extract features for training set
    # load_cached_data=False ensures we actually run the extraction logic
    X, y, ids = extractor.get_features(dataset_type="train", load_cached_data=False)

    print(f"Extracted features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Validation
    assert X.shape[0] == 2, "Expected 2 rows in mini train set"
    assert not X.isnull().values.any(), "Features contain NaNs"
    assert "sensor_1_mean" in X.columns, "Expected feature 'sensor_1_mean' missing"
    print("Tabular feature extraction verified.")
    return X, y


def demo_spectrogram_processing():
    print("\n=== Demo: Spectrogram Generation (Branch B) ===")
    generator = SpectrogramGenerator()

    # Generate spectrograms for training set
    X, y, ids = generator.get_dataset(dataset_type="train", load_cached_data=False)

    print(f"Spectrogram tensor shape: {X.shape}")

    # Validation
    # Shape should be (N, 10, 128, 256) based on Config
    assert X.shape == (2, 10, 128, 256), f"Unexpected shape {X.shape}"
    assert y is not None, "Targets should not be None for train set"
    assert len(y) == 2, "Target length mismatch"
    print("Spectrogram generation verified.")
    return X, y


def demo_lgbm_training(X, y):
    print("\n=== Demo: LightGBM Training ===")
    trainer = LGBMTrainer()

    # Split mini train into train/val for the trainer demo
    # Since we only have 2 samples, we'll just duplicate them for this specific unit test
    # to avoid empty dataset errors in LGBM
    X_train, y_train = X, y
    X_val, y_val = X, y

    model, best_score = trainer.train(X_train, y_train, X_val, y_val, fold=0)

    # Predict
    preds = trainer.predict(X_val, fold=0)

    # Validation
    assert len(preds) == len(X_val), "Prediction length mismatch"
    assert best_score >= 0, "MAE score should be non-negative"
    print("LightGBM training and prediction verified.")


def demo_cnn_training(X_spec, y_spec):
    print("\n=== Demo: CNN Training ===")

    # Create DataLoaders
    # y_spec is already scaled (log1p) by the generator if configured
    dataset = SeismicCNNDataset(X_spec, y_spec, is_train=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    trainer = CNNTrainer()

    # Train for 1 epoch (configured in overrides)
    val_mae = trainer.train(loader, loader, fold=0)

    # Predict
    preds = trainer.predict(loader, fold=0)

    # Validation
    assert len(preds) == len(X_spec), "Prediction length mismatch"
    assert val_mae >= 0, "Validation MAE should be non-negative"
    print("CNN training and prediction verified.")


def demo_stacking():
    print("\n=== Demo: Stacking (Ridge Regression) ===")
    stacker = RidgeStacker()

    # Dummy predictions for 4 samples
    preds_lgbm = np.array([100, 200, 300, 400])
    preds_cnn = np.array([110, 190, 310, 410])
    y_true = np.array([105, 195, 305, 405])

    X_meta = np.column_stack([preds_lgbm, preds_cnn])

    # Fit
    stacker.fit(X_meta, y_true)

    # Predict
    final_preds = stacker.predict(X_meta)

    # Validation
    assert len(final_preds) == 4
    print("Stacking verified.")


def demo_full_pipeline():
    print("\n=== Demo: Full Cross-Validation Pipeline ===")

    # Instantiate CrossValidator
    cv = CrossValidator()

    # Run pipeline
    # This will load data (using our mini metadata), run 2-fold CV (on 4 samples total),
    # train LGBM and CNN, stack them, and generate submission for the 2 test samples.
    cv.run()

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df_sub.head())

    assert len(df_sub) == 2, "Submission should have 2 rows (from mini test set)"
    assert "segment_id" in df_sub.columns
    assert "time_to_eruption" in df_sub.columns

    print("Full pipeline verified successfully.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Create Mini Metadata & Configure
    train_path, val_path, test_path = create_mini_metadata()
    configure_environment(train_path, val_path, test_path)

    # 3. Run Component Demos
    # Tabular
    X_tab, y_tab = demo_tabular_features()

    # Spectrogram
    X_spec, y_spec = demo_spectrogram_processing()

    # Models
    demo_lgbm_training(X_tab, y_tab)
    demo_cnn_training(X_spec, y_spec)

    # Stacking
    demo_stacking()

    # 4. Run Full Pipeline
    # We clear the cache first to ensure the pipeline runs data loading from scratch
    # (or we can let it use the cache we just generated, which is faster)
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
