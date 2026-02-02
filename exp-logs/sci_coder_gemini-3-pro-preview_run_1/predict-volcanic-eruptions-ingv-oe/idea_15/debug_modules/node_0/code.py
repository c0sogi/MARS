import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device, mae_score
from library.feature_engineering import FeatureEngineer
from library.dataset import VolcanoDataset
from library.model_vision import EfficientNetFiLM
from library.model_tabular import train_lgbm_fold, predict_lgbm
from library.runner import (
    run_tabular_cv,
    run_vision_cv,
    train_meta_learner,
    generate_submission,
)

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    print("Setting up demonstration environment...")

    # Define temporary directories
    DEMO_DIR = "./working/demo_pipeline"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_WORK_DIR = os.path.join(DEMO_DIR, "working")
    DEMO_SUB_DIR = os.path.join(DEMO_DIR, "submission")

    for d in [DEMO_META_DIR, DEMO_WORK_DIR, DEMO_SUB_DIR]:
        os.makedirs(d, exist_ok=True)

    # Patch Config for speed and isolation
    Config.METADATA_DIR = DEMO_META_DIR
    Config.WORKING_DIR = DEMO_WORK_DIR
    Config.SUBMISSION_DIR = DEMO_SUB_DIR
    Config.N_FOLDS = 2
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.LGB_PARAMS["n_estimators"] = 10
    Config.LGB_PARAMS["early_stopping_rounds"] = 5
    Config.LGB_PARAMS["verbosity"] = -1

    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Prepare Mini-Dataset (Subsetting Metadata)
    # ---------------------------------------------------------
    print("Creating mini-dataset metadata...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train.csv")
    orig_test_meta = pd.read_csv("./metadata/test.csv")

    # Sample subset (20 train, 10 test) to ensure speed
    mini_train = orig_train_meta.head(20).copy()
    mini_test = orig_test_meta.head(10).copy()

    # Save to demo metadata directory
    mini_train_path = os.path.join(DEMO_META_DIR, "train.csv")
    mini_test_path = os.path.join(DEMO_META_DIR, "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Mini-train saved: {len(mini_train)} rows")
    print(f"Mini-test saved: {len(mini_test)} rows")

    # ---------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # ---------------------------------------------------------
    print("\n--- Demonstrating Feature Engineering ---")

    fe = FeatureEngineer()

    # Process Training Data
    print("Processing training data...")
    df_train_feats = fe.process_dataset(
        mini_train_path, output_dir_name="train_data", load_cached_data=False
    )

    # Process Test Data
    print("Processing test data...")
    df_test_feats = fe.process_dataset(
        mini_test_path, output_dir_name="test_data", load_cached_data=False
    )

    # Assertions
    assert not df_train_feats.empty, "Training features DF is empty"
    assert not df_test_feats.empty, "Test features DF is empty"
    assert (
        "scalar_sensor_1_log_energy" in df_train_feats.columns
    ), "Scalar features missing"

    train_spec_dir = os.path.join(DEMO_WORK_DIR, "train_data", "spectrograms")
    test_spec_dir = os.path.join(DEMO_WORK_DIR, "test_data", "spectrograms")

    # Check if spectrograms exist
    sample_seg_id = mini_train.iloc[0]["segment_id"]
    expected_spec_path = os.path.join(train_spec_dir, f"{sample_seg_id}.npy")
    assert os.path.exists(
        expected_spec_path
    ), f"Spectrogram not found at {expected_spec_path}"

    print("Feature Engineering verified successfully.")

    # ---------------------------------------------------------
    # 4. Dataset & DataLoader Demonstration
    # ---------------------------------------------------------
    print("\n--- Demonstrating Dataset & DataLoader ---")

    # Instantiate Dataset
    ds_train = VolcanoDataset(df_train_feats, train_spec_dir, mode="train")

    # Check length
    assert (
        len(ds_train) == 20
    ), f"Dataset length mismatch. Expected 20, got {len(ds_train)}"

    # Check item retrieval
    spec, scalar, target = ds_train[0]

    print(f"Spectrogram Shape: {spec.shape}")
    print(f"Scalar Shape: {scalar.shape}")
    print(f"Target: {target}")

    # Assertions for shapes
    # Spec: (10 sensors, 256 freq, 256 time)
    assert spec.shape == (10, 256, 256), f"Incorrect spectrogram shape: {spec.shape}"
    # Target: scalar tensor
    assert target.ndim == 0, "Target should be a scalar tensor"

    # Create DataLoader
    train_loader = DataLoader(ds_train, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch_spec, batch_scalar, batch_target = next(iter(train_loader))

    assert batch_spec.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("Dataset and DataLoader verified successfully.")

    # ---------------------------------------------------------
    # 5. Vision Model Demonstration
    # ---------------------------------------------------------
    print("\n--- Demonstrating Vision Model (EfficientNetFiLM) ---")

    device = get_device()
    scalar_dim = len(ds_train.scalar_cols)

    # Instantiate Model
    model = EfficientNetFiLM(scalar_input_dim=scalar_dim).to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        batch_spec = batch_spec.to(device)
        batch_scalar = batch_scalar.to(device)
        output = model(batch_spec, batch_scalar)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch. Expected ({Config.BATCH_SIZE}, 1)"
    print("Vision Model forward pass verified successfully.")

    # ---------------------------------------------------------
    # 6. Tabular Model Demonstration
    # ---------------------------------------------------------
    print("\n--- Demonstrating Tabular Model (LightGBM) ---")

    # Prepare features
    exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df_train_feats.columns if c not in exclude_cols]
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train_feats[c])
    ]

    # Split for demo
    train_sub = df_train_feats.iloc[:15]
    val_sub = df_train_feats.iloc[15:]

    # Train
    lgb_model, val_preds = train_lgbm_fold(train_sub, val_sub, feature_cols)

    # Assertions
    assert len(val_preds) == 5, "Validation predictions length mismatch"
    assert isinstance(val_preds, np.ndarray), "Predictions should be numpy array"

    # Predict on test
    test_preds = predict_lgbm(lgb_model, df_test_feats, feature_cols)
    assert len(test_preds) == 10, "Test predictions length mismatch"

    print("Tabular Model training and prediction verified successfully.")

    # ---------------------------------------------------------
    # 7. Full Pipeline Integration (Runner)
    # ---------------------------------------------------------
    print("\n--- Running Full Integration Pipeline ---")

    # 1. Tabular CV
    oof_tab, tab_models, tab_feats = run_tabular_cv(
        df_train_feats, n_folds=Config.N_FOLDS
    )

    # 2. Vision CV
    oof_vis, vis_model_paths, vis_scalar_stats = run_vision_cv(
        df_train_feats, train_spec_dir, n_folds=Config.N_FOLDS
    )

    # 3. Meta Learner
    y_true = df_train_feats["time_to_eruption"].values
    meta_model = train_meta_learner(oof_tab, oof_vis, y_true)

    # 4. Generate Submission
    generate_submission(
        df_test_feats,
        test_spec_dir,
        tab_models,
        tab_feats,
        vis_model_paths,
        vis_scalar_stats,
        meta_model,
    )

    # Final Verification
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    assert (
        len(df_sub) == 10
    ), f"Submission length mismatch. Expected 10, got {len(df_sub)}"
    assert "segment_id" in df_sub.columns and "time_to_eruption" in df_sub.columns

    print("\nSUCCESS: Full pipeline executed and submission generated.")
