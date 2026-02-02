import os
import sys
import pandas as pd
import numpy as np
import warnings

# Import library modules
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.backbone import SemanticModel
from library.feature_extractor import FeatureEngineer
from library.regressor import LGBMRanker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting DASAR Pipeline Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # --------------------------------------------------------------------------
    print("\n[1/6] Configuring environment for demo...")

    # Override Config parameters to run a fast, small-scale demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 notebooks per dataset
    Config.BACKBONE_EPOCHS = 1  # Only 1 epoch for fine-tuning
    Config.BACKBONE_BATCH_SIZE = 4  # Small batch size
    Config.NUM_BOOST_ROUND = 10  # Few boosting rounds for LightGBM
    Config.EARLY_STOPPING_ROUNDS = 5

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Semantic Backbone Fine-Tuning
    # --------------------------------------------------------------------------
    print("\n[2/6] Fine-tuning Semantic Backbone...")

    # Initialize the model
    # Note: We use the default pretrained model defined in Config
    semantic_model = SemanticModel()

    # Define a specific output path for this demo
    demo_model_path = os.path.join(Config.WORKING_DIR, "demo_fine_tuned_model")

    # Run fine-tuning
    # We explicitly pass parameters to override defaults that were bound at import time
    semantic_model.fine_tune(
        train_metadata_path=Config.TRAIN_METADATA_PATH,
        output_path=demo_model_path,
        batch_size=Config.BACKBONE_BATCH_SIZE,
        epochs=Config.BACKBONE_EPOCHS,
        learning_rate=2e-5,
        debug=Config.DEBUG,
        load_cached_data=False,  # Force regeneration of pairs for the demo
    )

    # Verify model was saved
    assert os.path.exists(
        demo_model_path
    ), "Fine-tuned model directory was not created."
    print("Backbone fine-tuning successful.")

    # --------------------------------------------------------------------------
    # 3. Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[3/6] Extracting Features...")

    engineer = FeatureEngineer(semantic_model)

    # A. Extract Training Features
    print("-> Processing Training Set...")
    df_train_feats = engineer.extract_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        cache_name="demo_train_features",
        load_cached_data=False,
        debug=Config.DEBUG,
        batch_size=10,
    )

    # Validation
    assert not df_train_feats.empty, "Training features DataFrame is empty."
    assert (
        "target" in df_train_feats.columns
    ), "Target column missing from training features."
    assert "sim_max" in df_train_feats.columns, "Feature 'sim_max' missing."

    # B. Extract Validation Features
    print("-> Processing Validation Set...")
    df_val_feats = engineer.extract_features(
        metadata_path=Config.VAL_METADATA_PATH,
        mode="train",  # Use 'train' mode to calculate targets for evaluation
        cache_name="demo_val_features",
        load_cached_data=False,
        debug=Config.DEBUG,
        batch_size=10,
    )
    assert not df_val_feats.empty, "Validation features DataFrame is empty."

    # C. Extract Test Features
    print("-> Processing Test Set...")
    df_test_feats = engineer.extract_features(
        metadata_path=Config.TEST_METADATA_PATH,
        mode="test",  # Inference mode (no targets)
        cache_name="demo_test_features",
        load_cached_data=False,
        debug=Config.DEBUG,
        batch_size=10,
    )
    assert not df_test_feats.empty, "Test features DataFrame is empty."
    assert (
        "target" not in df_test_feats.columns
    ), "Target column should not exist in test features."

    print(
        f"Feature extraction complete. Train shape: {df_train_feats.shape}, Test shape: {df_test_feats.shape}"
    )

    # --------------------------------------------------------------------------
    # 4. Regressor Training (LightGBM)
    # --------------------------------------------------------------------------
    print("\n[4/6] Training LightGBM Ranker...")

    ranker = LGBMRanker()

    # Train the model
    model = ranker.train(df_train_feats, df_val_feats)

    assert model is not None, "Model training returned None."
    print("Regressor training successful.")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("\n[5/6] Generating Submission...")

    # Predict on test set
    test_preds = ranker.predict(df_test_feats)

    # Validate predictions
    assert len(test_preds) == len(
        df_test_feats
    ), "Number of predictions does not match number of test rows."
    assert np.all(
        (test_preds >= 0) & (test_preds <= 1.0)
    ), "Predictions contain values outside [0, 1] range (expected normalized ranks)."

    # Generate submission file
    demo_submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Temporarily override the config path to save to our demo location
    original_sub_path = Config.SUBMISSION_FILE_PATH
    Config.SUBMISSION_FILE_PATH = demo_submission_path

    ranker.generate_submission(
        df_test_feats, test_preds, test_metadata_path=Config.TEST_METADATA_PATH
    )

    # Restore config
    Config.SUBMISSION_FILE_PATH = original_sub_path

    # Verify submission file
    assert os.path.exists(demo_submission_path), "Submission file was not created."
    df_sub = pd.read_csv(demo_submission_path)
    print(f"Submission generated at {demo_submission_path}")
    print(f"Submission rows: {len(df_sub)}")
    print("Sample submission row:")
    print(df_sub.head(1))

    # --------------------------------------------------------------------------
    # 6. Logic Validation (Kendall Tau Metric)
    # --------------------------------------------------------------------------
    print("\n[6/6] Validating Metric Logic on Validation Set...")

    # Predict on validation set
    val_preds = ranker.predict(df_val_feats)

    # Generate a submission-like DataFrame for the validation set
    # We reuse generate_submission by pointing it to a temp file and the val metadata
    temp_val_sub_path = os.path.join(Config.WORKING_DIR, "temp_val_submission.csv")
    Config.SUBMISSION_FILE_PATH = temp_val_sub_path

    ranker.generate_submission(
        df_val_feats, val_preds, test_metadata_path=Config.VAL_METADATA_PATH
    )
    Config.SUBMISSION_FILE_PATH = original_sub_path  # Restore again

    # Load the predicted orders
    df_val_pred_orders = pd.read_csv(temp_val_sub_path)

    # Load ground truth
    df_val_truth = pd.read_csv(Config.VAL_METADATA_PATH)

    # Filter ground truth to only include the notebooks we processed in debug mode
    processed_val_ids = df_val_feats["id"].unique()
    df_val_truth_filtered = df_val_truth[df_val_truth["id"].isin(processed_val_ids)]

    # Compute Kendall Tau
    kt_score = compute_kendall_tau(df_val_truth_filtered, df_val_pred_orders)

    print(f"Kendall Tau Correlation on Validation Subset: {kt_score:.4f}")

    # Assert reasonable range
    assert (
        -1.0 <= kt_score <= 1.0
    ), f"Kendall Tau score {kt_score} is out of valid range [-1, 1]."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
