import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library import config, utils, feature_engineering, rf_module, nn_module

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set reproducible seed
    utils.set_seed(42)

    # Override configuration parameters for speed
    # We use a small sample size to ensure the pipeline runs quickly (within seconds/minutes)
    config.DATA_SAMPLE_SIZE = 50

    # Set up a specific working directory for this demo
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = config.WORKING_DIR
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Adjust Model Hyperparameters for the demo
    # Random Forest: Fewer trees, single job to avoid overhead on small data
    config.RF_PARAMS["n_estimators"] = 10
    config.RF_PARAMS["n_jobs"] = 1

    # Neural Network: Minimal epochs and batch size
    config.MLP_EPOCHS = 2
    config.MLP_BATCH_SIZE = 8
    config.MLP_PATIENCE = 1  # Stop early if needed

    print(f"    Working Directory: {config.WORKING_DIR}")
    print(f"    Data Sample Size: {config.DATA_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[2] executing Feature Engineering Pipeline...")

    # Instantiate the pipeline
    # Note: The pipeline uses the modified config.CACHE_DIR for storing artifacts
    pipeline = feature_engineering.FeaturePipeline()

    # Run the pipeline
    # load_cached_data=False forces re-computation to demonstrate the logic
    rf_data, mlp_data = pipeline.run(load_cached_data=False)

    # Validation: Verify data structures and shapes
    print("    Validating feature engineering outputs...")

    # RF Data Checks
    assert "X_train" in rf_data and "y_train" in rf_data
    assert (
        len(rf_data["X_train"]) == config.DATA_SAMPLE_SIZE
    ), f"Expected {config.DATA_SAMPLE_SIZE} training samples, got {len(rf_data['X_train'])}"
    assert rf_data["X_train"].shape[0] == rf_data["y_train"].shape[0]

    # MLP Data Checks
    assert "meta_train" in mlp_data and "title_train" in mlp_data
    assert mlp_data["meta_train"].shape[0] == config.DATA_SAMPLE_SIZE
    # Check embedding dimensions (SBERT default is 384)
    assert mlp_data["title_train"].shape[1] == 384

    print("    Feature engineering validation passed.")

    # -------------------------------------------------------------------------
    # 3. Random Forest Stream
    # -------------------------------------------------------------------------
    print("\n[3] Executing Random Forest Pipeline...")

    # Train and Predict
    rf_val_preds, rf_test_preds, rf_model = rf_module.run_rf_pipeline(rf_data)

    # Validation
    print("    Validating Random Forest outputs...")
    assert len(rf_val_preds) == len(rf_data["y_val"])
    assert len(rf_test_preds) == len(rf_data["X_test"])
    # Check probability range
    assert np.all((rf_test_preds >= 0) & (rf_test_preds <= 1))
    print("    Random Forest validation passed.")

    # -------------------------------------------------------------------------
    # 4. Neural Network Stream
    # -------------------------------------------------------------------------
    print("\n[4] Executing Neural Network Pipeline...")

    # Train and Predict
    nn_val_preds, nn_test_preds, nn_model = nn_module.run_nn_pipeline(mlp_data)

    # Validation
    print("    Validating Neural Network outputs...")
    assert len(nn_val_preds) == len(mlp_data["y_val"])
    assert len(nn_test_preds) == mlp_data["meta_test"].shape[0]
    # Check probability range
    assert np.all((nn_test_preds >= 0) & (nn_test_preds <= 1))
    print("    Neural Network validation passed.")

    # -------------------------------------------------------------------------
    # 5. Ensemble & Submission
    # -------------------------------------------------------------------------
    print("\n[5] Generating Ensemble Submission...")

    # Calculate weighted average
    print(f"    Weights -> RF: {config.WEIGHT_RF}, MLP: {config.WEIGHT_MLP}")
    final_preds = (rf_test_preds * config.WEIGHT_RF) + (
        nn_test_preds * config.WEIGHT_MLP
    )

    # Load test set metadata to get request_ids
    # We must respect the sample size used in the pipeline
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    if config.DATA_SAMPLE_SIZE:
        df_test = df_test.head(config.DATA_SAMPLE_SIZE)

    # Construct submission DataFrame
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": final_preds}
    )

    # Save submission
    output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(output_path, index=False)

    # Validation
    print(f"    Submission saved to: {output_path}")
    print("    Sample output:")
    print(submission.head())

    assert len(submission) == config.DATA_SAMPLE_SIZE
    assert submission["requester_received_pizza"].dtype == float

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
