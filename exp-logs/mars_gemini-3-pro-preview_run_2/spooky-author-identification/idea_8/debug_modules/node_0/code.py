import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_factory import DataManager
from library.classical_models import ClassicalModels
from library.training_engine import NeuralTrainer
from library.meta_learner import MetaLearner


def run_demo():
    print("==== STARTING PIPELINE DEMONSTRATION ====")

    # 1. Configure for Speed (Override Defaults)
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.N_FOLDS = 2  # Reduce folds to 2 for speed
    Config.EPOCHS = 1  # Reduce epochs to 1
    Config.TRAIN_BATCH_SIZE = 32  # Increase batch size for A100 efficiency
    Config.VALID_BATCH_SIZE = 64
    Config.PATIENCE = 1  # Aggressive early stopping
    Config.SVD_COMPONENTS = 50  # Reduce SVD dimensions
    # Use roberta-base instead of large for faster demo execution
    Config.MODEL_BACKBONES = ["roberta-base"]

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 2. Data Management & Feature Engineering
    print("\n[2] Verifying DataManager and Feature Engineering...")

    # Load Metadata
    train_df, val_df, test_df = DataManager.load_metadata()
    print(
        f"   Metadata Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    # Validate Metadata
    assert len(train_df) > 0, "Training metadata is empty"
    assert len(test_df) > 0, "Test metadata is empty"

    # Generate/Load Features (Force re-computation with load_cached_data=False to verify logic)
    print("   Computing Style Features...")
    train_style, val_style, test_style = DataManager.get_style_features(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert (
        train_style.shape[1] == 13
    ), f"Expected 13 style features, got {train_style.shape[1]}"

    print("   Computing TF-IDF Features...")
    train_tfidf, val_tfidf, test_tfidf = DataManager.get_tfidf_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    print("   Computing SVD Features...")
    train_svd, val_svd, test_svd = DataManager.get_svd_features(
        train_tfidf, val_tfidf, test_tfidf, load_cached_data=False
    )
    assert train_svd.shape[1] == Config.SVD_COMPONENTS, "SVD component mismatch"

    # 3. Classical Models Pipeline
    print("\n[3] Running Classical Models (LR, NB, XGB)...")
    classical_mgr = ClassicalModels()

    # Run CV for classical models
    # This returns a dictionary with 'oof' and 'test' predictions for each model
    classical_results = classical_mgr.run_classical_cv(load_cached_data=False)

    # Verify results
    expected_models = ["lr", "nb", "xgb"]
    total_train_samples = len(train_df) + len(val_df)

    for model_name in expected_models:
        assert model_name in classical_results, f"Missing result for {model_name}"
        oof_shape = classical_results[model_name]["oof"].shape
        test_shape = classical_results[model_name]["test"].shape

        print(f"   {model_name.upper()} -> OOF: {oof_shape}, Test: {test_shape}")

        assert oof_shape == (
            total_train_samples,
            Config.NUM_CLASSES,
        ), f"{model_name} OOF shape mismatch"
        assert test_shape == (
            len(test_df),
            Config.NUM_CLASSES,
        ), f"{model_name} Test shape mismatch"

    # 4. Neural Network Pipeline
    print("\n[4] Running Neural Network Pipeline...")
    neural_trainer = NeuralTrainer()

    # We use the first backbone defined in our override
    backbone_name = Config.MODEL_BACKBONES[0]
    print(f"   Training StylometricFusionModel with backbone: {backbone_name}")

    oof_neural, test_neural = neural_trainer.run_neural_cv(
        backbone_name, load_cached_data=False
    )

    print(f"   Neural -> OOF: {oof_neural.shape}, Test: {test_neural.shape}")
    assert oof_neural.shape == (total_train_samples, Config.NUM_CLASSES)
    assert test_neural.shape == (len(test_df), Config.NUM_CLASSES)

    # 5. Meta-Learner (Stacking)
    print("\n[5] Training Meta-Learner (Stacking)...")

    # Prepare dictionaries for the meta-learner
    oof_dict = {
        "lr": classical_results["lr"]["oof"],
        "nb": classical_results["nb"]["oof"],
        "xgb": classical_results["xgb"]["oof"],
        backbone_name: oof_neural,
    }

    test_dict = {
        "lr": classical_results["lr"]["test"],
        "nb": classical_results["nb"]["test"],
        "xgb": classical_results["xgb"]["test"],
        backbone_name: test_neural,
    }

    meta_learner = MetaLearner()
    final_preds = meta_learner.train_meta_learner(oof_dict, test_dict)

    # 6. Final Validation
    print("\n[6] Validating Submission...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission File Loaded: {submission_df.shape}")

    # Check columns
    expected_cols = ["id", "EAP", "HPL", "MWS"]
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    # Check ID consistency
    assert submission_df["id"].equals(
        test_df["id"]
    ), "Submission IDs do not match Test IDs"

    # Check probability sums (should be close to 1, though clipping might alter slightly)
    row_sums = submission_df[["EAP", "HPL", "MWS"]].sum(axis=1)
    mean_sum = row_sums.mean()
    print(f"   Mean Probability Sum: {mean_sum:.4f}")
    assert 0.99 < mean_sum < 1.01, "Probabilities do not sum to approx 1.0"

    print("\n==== DEMONSTRATION COMPLETED SUCCESSFULLY ====")


if __name__ == "__main__":
    run_demo()
