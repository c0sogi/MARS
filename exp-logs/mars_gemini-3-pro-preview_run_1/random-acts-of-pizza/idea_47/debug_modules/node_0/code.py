import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Setup Temporary Environment and Subset Data
    # ---------------------------------------------------------
    print("\n[1] Setting up temporary environment and subsetting data...")

    # Define paths
    original_metadata_dir = "./metadata"
    temp_dir = "./working/demo_temp"
    temp_metadata_dir = os.path.join(temp_dir, "metadata")
    temp_working_dir = os.path.join(temp_dir, "working")
    temp_submission_dir = os.path.join(temp_dir, "submission")

    os.makedirs(temp_metadata_dir, exist_ok=True)
    os.makedirs(temp_working_dir, exist_ok=True)
    os.makedirs(temp_submission_dir, exist_ok=True)

    # Subset size for speed optimization
    SUBSET_SIZE = 50

    # Create subset CSVs
    for filename in ["train.csv", "val.csv", "test.csv"]:
        src_path = os.path.join(original_metadata_dir, filename)
        dst_path = os.path.join(temp_metadata_dir, filename)

        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Take top N rows
            df_subset = df.head(SUBSET_SIZE)
            df_subset.to_csv(dst_path, index=False)
            print(f"    Created subset of {filename} with {len(df_subset)} rows.")
        else:
            raise FileNotFoundError(f"Original file {src_path} not found.")

    # ---------------------------------------------------------
    # 2. Patch Configuration (BEFORE importing other library modules)
    # ---------------------------------------------------------
    print("\n[2] Patching library configuration for speed...")

    import library.config as config

    # Patch Paths to use our temporary subset data
    config.METADATA_DIR = temp_metadata_dir
    config.WORKING_DIR = temp_working_dir
    config.SUBMISSION_DIR = temp_submission_dir
    config.CACHE_DIR = os.path.join(temp_working_dir, "cache")

    config.TRAIN_PATH = os.path.join(temp_metadata_dir, "train.csv")
    config.VAL_PATH = os.path.join(temp_metadata_dir, "val.csv")
    config.TEST_PATH = os.path.join(temp_metadata_dir, "test.csv")
    config.OUTPUT_SUBMISSION_PATH = os.path.join(temp_submission_dir, "submission.csv")

    # Ensure directories exist (since we changed the paths in config)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Patch Hyperparameters for Speed
    config.RF_N_ESTIMATORS = 10
    config.MLP_NUM_EPOCHS = 2
    config.MLP_BATCH_SIZE = 16
    config.MLP_HIDDEN_DIM = 64  # Smaller model

    print("    Configuration patched successfully.")

    # ---------------------------------------------------------
    # 3. Import Library Modules (AFTER patching config)
    # ---------------------------------------------------------
    print("\n[3] Importing library modules...")
    # Importing these now ensures they pick up the modified config values
    import library.utils as utils
    import library.data_loader as data_loader
    import library.feature_engineering as fe
    import library.dataset as dataset
    import library.models as models
    import library.train as train_module
    import library.predict as predict_module

    # ---------------------------------------------------------
    # 4. Demonstrate Utils
    # ---------------------------------------------------------
    print("\n[4] Testing library.utils...")

    # Test seed_everything
    utils.seed_everything(42)
    print("    seed_everything executed.")

    # Test pickle operations
    dummy_data = {"a": 1, "b": 2}
    pickle_path = os.path.join(temp_working_dir, "test.pkl")
    utils.save_pickle(dummy_data, pickle_path)
    loaded_data = utils.load_pickle(pickle_path)
    assert dummy_data == loaded_data, "Pickle save/load failed."
    print("    Pickle save/load verified.")

    # ---------------------------------------------------------
    # 5. Demonstrate Data Loader
    # ---------------------------------------------------------
    print("\n[5] Testing library.data_loader...")

    # Force reload from source (ignore cache for this test) to verify loading logic
    train_df, val_df, test_df = data_loader.load_dataset(load_cached_data=False)

    assert len(train_df) == SUBSET_SIZE, f"Train DF length mismatch: {len(train_df)}"
    assert len(val_df) == SUBSET_SIZE, f"Val DF length mismatch: {len(val_df)}"
    assert len(test_df) == SUBSET_SIZE, f"Test DF length mismatch: {len(test_df)}"
    assert "requester_subreddits_at_request" in train_df.columns, "Column missing."
    # Verify list conversion
    assert isinstance(
        train_df.iloc[0]["requester_subreddits_at_request"], list
    ), "List conversion failed."

    print("    Data loaded and parsed correctly.")

    # ---------------------------------------------------------
    # 6. Demonstrate Feature Engineering
    # ---------------------------------------------------------
    print("\n[6] Testing library.feature_engineering...")

    pipeline = fe.FeaturePipeline()
    # Run pipeline (this will use the subsetted data loaded internally by the pipeline)
    # We set load_cached_data=False to force computation on our new subset
    rf_out, mlp_out = pipeline.run(load_cached_data=False)

    # Verify RF features
    assert rf_out["train_X"].shape[0] == SUBSET_SIZE, "RF Train X shape mismatch."
    assert rf_out["train_y"].shape[0] == SUBSET_SIZE, "RF Train y shape mismatch."

    # Verify MLP features
    assert mlp_out["train_title"].shape[0] == SUBSET_SIZE, "MLP Train title mismatch."
    assert mlp_out["train_meta"].shape[0] == SUBSET_SIZE, "MLP Train meta mismatch."

    print("    Feature pipeline executed successfully.")

    # ---------------------------------------------------------
    # 7. Demonstrate Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n[7] Testing library.dataset...")

    # Create dataloaders using the cached features from step 6 (load_cached_data=True now works)
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        load_cached_data=True, batch_size=config.MLP_BATCH_SIZE
    )

    # Check batch structure
    batch = next(iter(train_loader))
    assert "title_emb" in batch
    assert "metadata" in batch
    assert "label" in batch
    assert batch["title_emb"].shape[0] <= config.MLP_BATCH_SIZE

    print("    DataLoaders created and batch verified.")

    # ---------------------------------------------------------
    # 8. Demonstrate Models
    # ---------------------------------------------------------
    print("\n[8] Testing library.models...")

    # Test RF Wrapper
    rf_model = models.InteractionRandomForest()
    rf_model.fit(rf_out["train_X"], rf_out["train_y"])
    probs = rf_model.predict_proba(rf_out["val_X"])
    assert probs.shape == (SUBSET_SIZE, 2), "RF output shape mismatch."
    print("    Random Forest instantiated and fit.")

    # Test MLP Architecture
    metadata_dim = batch["metadata"].shape[1]
    mlp_model = models.PizzaFiLMMLP(
        metadata_dim=metadata_dim, hidden_dim=config.MLP_HIDDEN_DIM
    ).to(config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = mlp_model(
            batch["title_emb"].to(config.DEVICE),
            batch["body_emb"].to(config.DEVICE),
            batch["history_emb"].to(config.DEVICE),
            batch["history_mask"].to(config.DEVICE),
            batch["centroid_emb"].to(config.DEVICE),
            batch["metadata"].to(config.DEVICE),
        )

    assert logits.shape == (
        batch["title_emb"].shape[0],
        1,
    ), "MLP output shape mismatch."
    print("    MLP instantiated and forward pass successful.")

    # ---------------------------------------------------------
    # 9. Demonstrate Training Functions
    # ---------------------------------------------------------
    print("\n[9] Testing library.train...")

    # Train RF
    print("    Training RF via train_rf()...")
    trained_rf = train_module.train_rf(load_cached_data=True)
    assert isinstance(trained_rf, models.InteractionRandomForest)

    # Train MLP
    print("    Training MLP via train_mlp()...")
    trained_mlp = train_module.train_mlp(
        load_cached_data=True, batch_size=config.MLP_BATCH_SIZE
    )
    assert isinstance(trained_mlp, models.PizzaFiLMMLP)

    print("    Training functions executed successfully.")

    # ---------------------------------------------------------
    # 10. Demonstrate Prediction Pipeline
    # ---------------------------------------------------------
    print("\n[10] Testing library.predict...")

    # Run full prediction pipeline
    predict_module.run_prediction(load_cached_data=True)

    # Verify submission file
    assert os.path.exists(config.OUTPUT_SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(config.OUTPUT_SUBMISSION_PATH)
    assert len(sub_df) == SUBSET_SIZE, "Submission row count mismatch."
    assert "request_id" in sub_df.columns, "request_id column missing."
    assert "requester_received_pizza" in sub_df.columns, "target column missing."

    print("    Prediction pipeline executed and submission generated.")

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    print("\n[11] Cleaning up...")
    try:
        shutil.rmtree(temp_dir)
        print("    Temporary directory removed.")
    except Exception as e:
        print(f"    Warning: Could not remove temp dir: {e}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
