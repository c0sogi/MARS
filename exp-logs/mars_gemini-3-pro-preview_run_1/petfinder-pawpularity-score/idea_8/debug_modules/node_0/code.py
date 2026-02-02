import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_array
from library.dataset import load_dataset, PawpularityDataset, get_transforms
from library.feature_extraction import extract_and_save_features
from library.feature_processing import get_expert_data
from library.level0_experts import Level0Trainer
from library.level1_meta import MetaLearner


def run_demo():
    print(">>> Starting Pawpularity Pipeline Demo")

    # =========================================================================
    # 1. Configuration Overrides for Speed
    # =========================================================================
    print("\n[1] Configuring environment for rapid demonstration...")
    # Enable Debug mode to use a small subset (e.g., 50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50

    # Reduce CV folds
    Config.N_FOLDS = 2

    # Simplify Hyperparameters for speed
    Config.RIDGE_ALPHAS = [1.0, 10.0]
    Config.KNN_NEIGHBORS = [5]
    Config.ET_N_ESTIMATORS = 10
    Config.META_N_ITER = 10

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, N_FOLDS=2")

    # =========================================================================
    # 2. Dataset & Transforms Demonstration
    # =========================================================================
    print("\n[2] Verifying Dataset and Transforms...")

    # Load metadata (Debug mode will load top 50 rows)
    df_train = load_dataset("train")
    assert (
        len(df_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(df_train)}"

    # Setup transforms
    transform = get_transforms(img_size=224)

    # Initialize Dataset
    dataset = PawpularityDataset(
        df_train, Config.INPUT_DIR, transform=transform, return_id=True
    )

    # Fetch one sample
    sample = dataset[0]

    # Verify keys and shapes
    assert "image" in sample
    assert "meta" in sample
    assert "target" in sample
    assert "id" in sample

    # Image should be (3, 224, 224)
    assert sample["image"].shape == (
        3,
        224,
        224,
    ), f"Image shape mismatch: {sample['image'].shape}"
    # Meta should be (12,)
    assert sample["meta"].shape == (12,), f"Meta shape mismatch: {sample['meta'].shape}"

    print("Dataset verification passed.")

    # =========================================================================
    # 3. Feature Extraction Demonstration
    # =========================================================================
    print("\n[3] Running Feature Extraction (CLIP Backbone)...")

    # We will only extract CLIP features to save time.
    # This function handles Train, Val, and Test splits internally if we call it for each.
    # However, get_expert_data calls it internally if cache is missing.
    # We'll explicitly run it for 'train' to demonstrate usage.

    clip_features_data = extract_and_save_features(
        model_name=Config.MODEL_CLIP,
        split="train",
        load_cached_data=False,  # Force run
        batch_size=16,
    )

    # Verify outputs
    feats = clip_features_data["features"]
    ids = clip_features_data["ids"]

    # CLIP Large output dim is 768 + N_prompts (13) = 781
    expected_dim = 768 + len(Config.AESTHETIC_PROMPTS)
    assert (
        feats.shape[1] == expected_dim
    ), f"Feature dim mismatch. Expected {expected_dim}, got {feats.shape[1]}"
    assert len(ids) == Config.DEBUG_SAMPLE_SIZE

    print(f"Feature extraction successful. Shape: {feats.shape}")

    # =========================================================================
    # 4. Feature Processing Demonstration
    # =========================================================================
    print("\n[4] Running Feature Processing (Linear Group)...")

    # This prepares data for Ridge/SVR (StandardScaler on concatenated vectors)
    # It will internally trigger feature extraction for val and test splits if missing.
    processed_data = get_expert_data(
        model_name=Config.MODEL_CLIP, expert_group="linear", load_cached_data=False
    )

    X_train = processed_data["X_train"]
    X_val = processed_data["X_val"]
    X_test = processed_data["X_test"]

    # Verify shapes
    # Train/Val split is roughly 80/20 of the debug sample size
    # Note: load_dataset("val") loads the validation csv which is distinct from train csv.
    # In debug mode, both load top N rows of their respective CSVs.
    assert X_train.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert X_val.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert X_test.shape[0] == Config.DEBUG_SAMPLE_SIZE

    # Dimension check: Embedding + Scores + Meta
    # CLIP Embedding (768) + Scores (13) + Meta (12) = 793
    # Note: The 'features' from CLIPWrapper already contain scores.
    # feature_processing._load_raw_data separates them.
    # Then _concat puts them back together + meta.
    # So total should be 768 + 13 + 12 = 793.
    expected_proc_dim = 768 + 13 + 12
    assert (
        X_train.shape[1] == expected_proc_dim
    ), f"Processed dim mismatch. Expected {expected_proc_dim}, got {X_train.shape[1]}"

    print("Feature processing successful.")

    # =========================================================================
    # 5. Level-0 Expert Training Demonstration
    # =========================================================================
    print("\n[5] Training Level-0 Experts...")

    # We will train two experts: Ridge and SVR, both using CLIP backbone.

    # Expert 1: Ridge
    trainer_ridge = Level0Trainer(backbone_name=Config.MODEL_CLIP, expert_name="ridge")
    res_ridge = trainer_ridge.run(load_cached_data=False)

    # Expert 2: SVR
    trainer_svr = Level0Trainer(backbone_name=Config.MODEL_CLIP, expert_name="svr")
    res_svr = trainer_svr.run(load_cached_data=False)

    # Verify outputs
    # OOF size should be len(Train) + len(Val) = 50 + 50 = 100
    expected_oof_len = Config.DEBUG_SAMPLE_SIZE * 2
    assert len(res_ridge["oof"]) == expected_oof_len
    assert len(res_svr["oof"]) == expected_oof_len

    print("Level-0 Experts trained successfully.")

    # =========================================================================
    # 6. Level-1 Meta Learner Demonstration
    # =========================================================================
    print("\n[6] Running Meta-Learner...")

    meta_learner = MetaLearner()

    # IMPORTANT: Restrict MetaLearner to only use the experts we just trained.
    # Otherwise, it will attempt to extract features for DINO/ConvNeXt and train KNN/ExtraTrees,
    # which would take too long for this demo.
    meta_learner.backbones = [Config.MODEL_CLIP]
    meta_learner.experts = ["ridge", "svr"]

    # Run the meta-learner pipeline
    final_rmse = meta_learner.run(load_cached_data=False)

    print(f"Meta-Learner finished with RMSE: {final_rmse}")

    # =========================================================================
    # 7. Final Submission Verification
    # =========================================================================
    print("\n[7] Verifying Submission File...")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file loaded. Shape: {sub_df.shape}")

        # Check columns
        assert "Id" in sub_df.columns
        assert "Pawpularity" in sub_df.columns

        # Check length (should match test debug size)
        assert len(sub_df) == Config.DEBUG_SAMPLE_SIZE

        # Check values are within reasonable range (0-100)
        # Note: Regressors can sometimes predict slightly outside bounds, but usually close.
        preds = sub_df["Pawpularity"]
        print(
            f"Prediction stats: Min={preds.min():.2f}, Max={preds.max():.2f}, Mean={preds.mean():.2f}"
        )
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
