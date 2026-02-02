import os
import pandas as pd
import numpy as np
import torch
import shutil
from library.config import Config
from library.utils import set_seed, kendall_tau_metric
from library.dataset import (
    create_relaxed_proximity_pairs,
    prepare_ranking_data,
    prepare_test_data,
)
from library.backbones import BackboneFineTuner
from library.features import DualViewFeatureExtractor
from library.regressor import RankRegressor, generate_submission


def main():
    print("=== DVSAR Pipeline Demo ===")

    # 1. Setup and Configuration Overrides
    # We override default hyperparameters to ensure the demo runs quickly (within minutes)
    print("Step 1: Configuring environment for fast execution...")
    set_seed(42)

    # Override Config parameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.WARMUP_STEPS = 5
    # LightGBM speed optimizations
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["min_child_samples"] = 5

    # Ensure working directory exists
    Config.setup()

    # 2. Data Preparation
    # We generate data for both Stage 1 (Contrastive) and Stage 2 (Regression)
    # Using debug=True limits processing to the first 100 notebooks
    print("\nStep 2: Preparing Datasets (Debug Mode)...")

    # Stage 1: Pairs for Contrastive Learning
    df_pairs = create_relaxed_proximity_pairs(
        Config.TRAIN_PATH, mode="train", debug=True, load_cached_data=False
    )
    assert not df_pairs.empty, "Training pairs DataFrame should not be empty."
    assert "markdown_text" in df_pairs.columns and "code_text" in df_pairs.columns
    print(f"  - Generated {len(df_pairs)} pairs for contrastive fine-tuning.")

    # Stage 2: Ranking Data for Regression
    df_rank_train = prepare_ranking_data(
        Config.TRAIN_PATH, mode="train", debug=True, load_cached_data=False
    )
    df_rank_val = prepare_ranking_data(
        Config.VAL_PATH, mode="val", debug=True, load_cached_data=False
    )
    assert not df_rank_train.empty, "Training ranking data should not be empty."
    assert "rank" in df_rank_train.columns
    print(f"  - Generated {len(df_rank_train)} training ranking samples.")
    print(f"  - Generated {len(df_rank_val)} validation ranking samples.")

    # Inference Data
    df_test_data = prepare_test_data(
        Config.TEST_PATH, debug=True, load_cached_data=False
    )
    assert not df_test_data.empty, "Test data should not be empty."
    print(f"  - Generated {len(df_test_data)} test samples.")

    # 3. Stage 1: Fine-Tuning Backbones
    # We fine-tune two separate models: one for text view, one for code view.
    print("\nStep 3: Fine-Tuning Backbones (Contrastive Learning)...")

    # Fine-tune Text Model
    print("  - Training Text-View Model...")
    text_tuner = BackboneFineTuner(Config.MODEL_TEXT, Config.TEXT_MODEL_SAVE_PATH)
    text_tuner.train(debug=True, load_cached_data=True, force_retrain=True)
    assert os.path.exists(Config.TEXT_MODEL_SAVE_PATH), "Text model failed to save."

    # Fine-tune Code Model
    print("  - Training Code-View Model...")
    code_tuner = BackboneFineTuner(Config.MODEL_CODE, Config.CODE_MODEL_SAVE_PATH)
    code_tuner.train(debug=True, load_cached_data=True, force_retrain=True)
    assert os.path.exists(Config.CODE_MODEL_SAVE_PATH), "Code model failed to save."

    # 4. Feature Extraction
    # Use the fine-tuned models to extract similarity features
    print("\nStep 4: Extracting Dual-View Features...")
    extractor = DualViewFeatureExtractor()

    # Extract for Train
    df_feats_train = extractor.extract_features(
        Config.TRAIN_PATH, mode="train", debug=True, load_cached_data=False
    )
    # Extract for Val
    df_feats_val = extractor.extract_features(
        Config.VAL_PATH, mode="val", debug=True, load_cached_data=False
    )
    # Extract for Test
    df_feats_test = extractor.extract_features(
        Config.TEST_PATH, mode="test", debug=True, load_cached_data=False
    )

    # Validation of feature structure
    required_cols = ["tv_sim_max", "cv_sim_max", "n_code", "md_len"]
    for col in required_cols:
        assert col in df_feats_train.columns, f"Missing feature {col} in train."
        assert col in df_feats_test.columns, f"Missing feature {col} in test."

    assert (
        "target" in df_feats_train.columns
    ), "Target column missing in train features."
    print(
        f"  - Features extracted: Train {df_feats_train.shape}, Val {df_feats_val.shape}, Test {df_feats_test.shape}"
    )

    # 5. Stage 2: Regression Training
    # Train LightGBM to predict relative rank
    print("\nStep 5: Training Rank Regressor (LightGBM)...")
    regressor = RankRegressor()
    regressor.train(df_feats_train, df_feats_val)

    model_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
    assert os.path.exists(model_path), "Regressor model file not found."
    print("  - Regressor trained and saved.")

    # 6. Submission Generation
    # Predict ranks and reconstruct cell order
    print("\nStep 6: Generating Submission...")
    generate_submission(regressor, df_feats_test, Config.TEST_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        "id",
        "cell_order",
    ], "Submission columns are incorrect."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check a sample row
    sample_order = df_sub.iloc[0]["cell_order"]
    assert (
        isinstance(sample_order, str) and len(sample_order) > 0
    ), "Invalid cell_order format."
    print(f"  - Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} rows.")

    # 7. Metric Logic Verification
    # Unit test for the Kendall Tau metric function provided in utils
    print("\nStep 7: Verifying Metric Logic...")

    # Case A: Perfect Match
    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    score_perfect = kendall_tau_metric(df_pred_perfect, df_gt)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected 1.0 for perfect match, got {score_perfect}"

    # Case B: Complete Inversion (n=3)
    # n=3, pairs=3. Inversion a b c -> c b a requires 3 swaps.
    # Score = 1 - 4 * (3 / (3*2)) = 1 - 2 = -1.0
    df_pred_inverse = pd.DataFrame({"id": ["nb1"], "cell_order": ["c b a"]})
    score_inverse = kendall_tau_metric(df_pred_inverse, df_gt)
    assert np.isclose(
        score_inverse, -1.0
    ), f"Expected -1.0 for inverse match, got {score_inverse}"

    print("  - Metric logic verified successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
