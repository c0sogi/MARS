import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import shutil

# Import provided library modules
import library.config
import library.dicom_feature_extractor
import library.data_processor
import library.model_zoo
import library.workflow


def create_mini_metadata(working_dir):
    """
    Creates a small subset of the metadata for demonstration purposes.
    """
    print("Creating mini-datasets for fast demonstration...")

    # Load original metadata
    train_df = pd.read_csv(library.config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(library.config.VAL_METADATA_PATH)
    test_df = pd.read_csv(library.config.TEST_METADATA_PATH)

    # Sample a few patients (ensure we get all rows for selected patients)
    train_patients = train_df["Patient"].unique()[:5]
    val_patients = val_df["Patient"].unique()[:2]
    # For test, we use the patients present in the test metadata
    test_patients = test_df["Patient"].unique()[:2]

    mini_train = train_df[train_df["Patient"].isin(train_patients)].copy()
    mini_val = val_df[val_df["Patient"].isin(val_patients)].copy()
    mini_test = test_df[test_df["Patient"].isin(test_patients)].copy()

    # Define paths
    mini_train_path = os.path.join(working_dir, "mini_train.csv")
    mini_val_path = os.path.join(working_dir, "mini_val.csv")
    mini_test_path = os.path.join(working_dir, "mini_test.csv")

    # Save
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Mini-datasets saved to {working_dir}")
    return mini_train_path, mini_val_path, mini_test_path


def patch_library_paths(train_path, val_path, test_path, cache_dir):
    """
    Patches the imported modules to use the mini-dataset paths and a temp cache dir.
    This is necessary because the modules import constants from config.py at load time.
    """
    print("Patching library modules to use mini-datasets...")

    # Patch dicom_feature_extractor
    library.dicom_feature_extractor.TRAIN_METADATA_PATH = train_path
    library.dicom_feature_extractor.VAL_METADATA_PATH = val_path
    library.dicom_feature_extractor.TEST_METADATA_PATH = test_path
    library.dicom_feature_extractor.CACHE_DIR = cache_dir

    # Patch data_processor
    library.data_processor.TRAIN_METADATA_PATH = train_path
    library.data_processor.VAL_METADATA_PATH = val_path
    library.data_processor.TEST_METADATA_PATH = test_path
    library.data_processor.CACHE_DIR = cache_dir

    # Patch model_zoo
    library.model_zoo.CACHE_DIR = cache_dir

    # Patch workflow (though it calls functions from above, good to be safe)
    # Workflow uses imports from other files, so patching those files above is key.
    pass


def verify_feature_extraction(train_path):
    """
    Demonstrates and verifies the VarianceWeightedExtractor.
    """
    print("\n=== Verifying Feature Extraction ===")

    # Initialize Extractor
    extractor = library.dicom_feature_extractor.VarianceWeightedExtractor(
        device=library.config.DEVICE
    )

    # Load mini metadata
    df = pd.read_csv(train_path)

    # Process just one patient manually to verify logic
    patient_id = df.iloc[0]["Patient"]
    dcm_dir = df.iloc[0]["dcm_path"]

    print(f"Extracting features for patient {patient_id}...")
    embedding = extractor.extract_patient_embedding(dcm_dir)

    # Verification
    assert isinstance(embedding, np.ndarray), "Embedding must be a numpy array"
    assert embedding.shape == (
        1280,
    ), f"Expected embedding shape (1280,), got {embedding.shape}"
    assert not np.isnan(embedding).any(), "Embedding contains NaNs"

    print("Feature extraction verification successful.")
    return extractor


def verify_data_processing(extractor, train_path, val_path, test_path):
    """
    Demonstrates and verifies the TabularPreprocessor.
    """
    print("\n=== Verifying Data Processing ===")

    # 1. Run extraction for all mini-sets (using the patched paths implicitly via run_extraction logic if we used it,
    # but here we call process_dataset directly to control inputs)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # We disable loading from cache to ensure we run the logic
    train_feats = extractor.process_dataset(
        train_df, "mini_train", load_cached_data=False
    )
    val_feats = extractor.process_dataset(val_df, "mini_val", load_cached_data=False)
    test_feats = extractor.process_dataset(test_df, "mini_test", load_cached_data=False)

    # 2. Initialize Preprocessor
    preprocessor = library.data_processor.TabularPreprocessor(
        pca_components=10
    )  # Reduced PCA for mini data

    # 3. Fit
    print("Fitting preprocessor...")
    preprocessor.fit(train_df, train_feats)

    # 4. Transform
    print("Transforming datasets...")
    X_fvc_train, X_unc_train, y_train, _ = preprocessor.transform(
        train_df, train_feats, is_test=False
    )
    X_fvc_test, X_unc_test, _, test_ids = preprocessor.transform(
        test_df, test_feats, is_test=True
    )

    # Verification
    # X_fvc structure: [Base, Time, PCA*Time]
    # Base = Num(3) + Cat(encoded) + PCA(10)
    # Time = 1
    # PCA*Time = 10
    # Total approx: 3 + (Sex+Smoking) + 10 + 1 + 10.
    # Sex=2, Smoking=3 (usually).

    print(f"X_fvc_train shape: {X_fvc_train.shape}")
    print(f"X_unc_train shape: {X_unc_train.shape}")

    assert X_fvc_train.shape[0] == len(train_df), "Row count mismatch in X_fvc_train"
    assert y_train.shape[0] == len(train_df), "Row count mismatch in y_train"
    assert X_fvc_test.shape[0] == len(test_df), "Row count mismatch in X_fvc_test"
    assert len(test_ids) == len(test_df), "Row count mismatch in test_ids"

    # Check for NaNs
    assert not np.isnan(X_fvc_train).any(), "X_fvc_train contains NaNs"

    print("Data processing verification successful.")
    return (
        X_fvc_train,
        y_train,
        X_unc_train,
        X_fvc_test,
        X_unc_test,
        test_ids,
        train_feats,
        val_feats,
        test_feats,
    )


def verify_model_training(data_tuple):
    """
    Demonstrates and verifies Model Zoo components.
    """
    print("\n=== Verifying Model Training ===")
    X_fvc_train, y_train, X_unc_train, X_fvc_test, X_unc_test, test_ids, _, _, _ = (
        data_tuple
    )

    # 1. Bagged Quantile Regressor
    print("Training BaggedQuantileRegressor (FVC)...")
    # Reduce n_estimators for speed
    fvc_model = library.model_zoo.BaggedQuantileRegressor(n_estimators=5, seed=42)
    fvc_model.fit(X_fvc_train, y_train)

    y_pred_train = fvc_model.predict(X_fvc_train)
    y_pred_test = fvc_model.predict(X_fvc_test)

    assert y_pred_train.shape == (len(X_fvc_train),), "Prediction shape mismatch"
    # FVC should be positive and reasonable
    assert (y_pred_train > 0).all(), "Predicted FVC should be positive"

    # 2. Bagged ElasticNet (Uncertainty)
    print("Training BaggedElasticNet (Uncertainty)...")
    residuals = np.abs(y_train - y_pred_train)
    unc_model = library.model_zoo.BaggedElasticNet(n_estimators=5, seed=42)
    unc_model.fit(X_unc_train, residuals)

    mad_pred = unc_model.predict(X_unc_test)
    sigma_pred = mad_pred * np.sqrt(2)

    assert (sigma_pred >= 0).all(), "Confidence should be non-negative"

    # 3. Metric
    score = library.model_zoo.laplace_log_likelihood(
        y_train, y_pred_train, sigma_pred[: len(y_train)]
    )  # Just using first N for dimension match demo
    print(f"Sample Metric Score: {score}")

    print("Model training verification successful.")


def verify_full_workflow():
    """
    Runs the full workflow using the patched paths.
    """
    print("\n=== Verifying Full Workflow ===")

    # We force load_cached_data=False to ensure the pipeline runs fully using our mini data
    # (Since we patched the paths, run_extraction will look at mini_train.csv etc.)

    try:
        library.workflow.run_workflow(load_cached_data=False)
    except Exception as e:
        print(f"Workflow failed with error: {e}")
        raise e

    # Check submission
    sub_path = library.config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file was not created"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in df_sub.columns for col in required_cols
    ), "Submission missing required columns"

    print("Full workflow verification successful.")


if __name__ == "__main__":
    # Setup directories
    working_dir = "./working/demo_run"
    cache_dir = os.path.join(working_dir, "cache")
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Create Mini Metadata
    mini_train_path, mini_val_path, mini_test_path = create_mini_metadata(working_dir)

    # 2. Patch Libraries
    patch_library_paths(mini_train_path, mini_val_path, mini_test_path, cache_dir)

    # 3. Verify Components
    extractor = verify_feature_extraction(mini_train_path)

    data_tuple = verify_data_processing(
        extractor, mini_train_path, mini_val_path, mini_test_path
    )

    verify_model_training(data_tuple)

    # 4. Verify Full Workflow
    verify_full_workflow()

    # Cleanup
    print("\nCleaning up temporary files...")
    # shutil.rmtree(working_dir) # Optional: comment out to inspect results
    print("Done.")
