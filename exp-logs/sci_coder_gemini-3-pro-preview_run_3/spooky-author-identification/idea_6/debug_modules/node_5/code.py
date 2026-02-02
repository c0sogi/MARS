import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.features import get_data
from library.stat_engine import StatisticalTrainer
from library.mlm_engine import run_mlm_pretraining
from library.neural_engine import NeuralTrainer
from library.ensemble import EnsembleOptimizer


def run_demo():
    print("=== Starting Author Identification Pipeline Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small subset for demo
    Config.N_FOLDS = 2  # Minimal folds for CV
    Config.EPOCHS = 1  # Single epoch for Neural training
    Config.MLM_EPOCHS = 1  # Single epoch for MLM
    Config.MLM_BATCH_SIZE = 4
    Config.BATCH_SIZE = 8

    # Use a lighter backbone for the demo to save time/memory if needed,
    # but we will stick to the config default (roberta/deberta) as they are standard.
    # We'll just use one backbone for the neural demo to save time.
    Config.MODEL_BACKBONES = ["roberta-base"]

    # Clean working directory for a fresh run (optional, but good for demo)
    # We only clean specific subfolders to avoid deleting pre-provided files if any
    if os.path.exists(Config.FEATURES_DIR):
        shutil.rmtree(Config.FEATURES_DIR)
    os.makedirs(Config.FEATURES_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Folds: {Config.N_FOLDS}")

    # 2. Feature Extraction (TF-IDF + Stylometric)
    print("\n[2] Running Feature Extraction...")
    # Force recompute to demonstrate the extraction logic
    X_train, y_train, X_val, y_val, X_test, test_ids = get_data(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Validations
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    assert X_train.shape[0] == Config.DEBUG_SAMPLE_SIZE, "Train sample size mismatch"
    assert X_val.shape[0] == Config.DEBUG_SAMPLE_SIZE, "Val sample size mismatch"
    assert X_test.shape[0] == Config.DEBUG_SAMPLE_SIZE, "Test sample size mismatch"
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature dimension mismatch between Train and Test"

    # 3. Statistical Model Training
    print("\n[3] Running Statistical Model (Logistic Regression)...")
    stat_trainer = StatisticalTrainer(n_folds=Config.N_FOLDS, random_state=Config.SEED)

    # Combine Train and Val manually to match the assertion expectation
    print("Combining Train and Val for CV...")
    X_train_full = scipy.sparse.vstack([X_train, X_val]).tocsr()
    y_train_full = np.concatenate([y_train, y_val])

    # Note: run_cv internally combines X_train and X_val to perform its own CV split
    # We pass the loaded data to avoid reloading
    stat_oof, stat_test_preds, stat_y_sorted, stat_ids = stat_trainer.run_cv(
        X_train=X_train_full,
        y_train=y_train_full,
        X_test=X_test,
        test_ids=test_ids,
        load_cached_data=False,
        debug=Config.DEBUG,
    )

    # Validations
    # Total samples for CV = Train + Val samples loaded
    expected_cv_samples = Config.DEBUG_SAMPLE_SIZE * 2
    assert stat_oof.shape == (
        expected_cv_samples,
        Config.NUM_CLASSES,
    ), "Statistical OOF shape incorrect"
    assert stat_test_preds.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        Config.NUM_CLASSES,
    ), "Statistical Test Preds shape incorrect"

    stat_loss = calculate_log_loss(stat_y_sorted, stat_oof)
    print(f"Statistical Model CV Log Loss: {stat_loss:.5f}")

    # 4. MLM Pretraining (Domain Adaptation)
    print("\n[4] Running MLM Pretraining...")
    # This will fine-tune the language model on the dataset text
    # We use the reduced MODEL_BACKBONES list
    mlm_paths = run_mlm_pretraining(
        model_backbones=Config.MODEL_BACKBONES,
        load_cached_data=False,
        debug=Config.DEBUG,
    )

    assert len(mlm_paths) > 0, "MLM training failed to return model paths"
    model_name = Config.MODEL_BACKBONES[0]
    assert model_name in mlm_paths, f"Path for {model_name} missing"
    print(f"MLM Model saved at: {mlm_paths[model_name]}")

    # 5. Neural Model Training
    print(f"\n[5] Running Neural Model Training ({model_name})...")
    neural_trainer = NeuralTrainer(model_name=model_name, n_folds=Config.N_FOLDS)

    # Run CV
    neural_oof, neural_test_preds, neural_y_sorted, neural_ids = neural_trainer.run_cv(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Validations
    assert neural_oof.shape == (
        expected_cv_samples,
        Config.NUM_CLASSES,
    ), "Neural OOF shape incorrect"
    assert neural_test_preds.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        Config.NUM_CLASSES,
    ), "Neural Test Preds shape incorrect"

    neural_loss = calculate_log_loss(neural_y_sorted, neural_oof)
    print(f"Neural Model CV Log Loss: {neural_loss:.5f}")

    # 6. Ensemble Optimization
    print("\n[6] Running Ensemble Optimization...")

    # Ensure targets match (they should, as both pipelines sort/combine data similarly,
    # but let's verify alignment just in case. In this specific library implementation,
    # stat_engine and neural_engine both load data using the same utils and sorting,
    # but stat_engine combines train+val explicitly. Neural engine loads train+val and concatenates.
    # We must ensure the y_true arrays are identical.)

    if not np.array_equal(stat_y_sorted, neural_y_sorted):
        print(
            "Warning: Target arrays differ. Aligning based on IDs is recommended in production."
        )
        # For this demo, we assume the data loading order is deterministic (fixed seed).
        # If they differ, we can't ensemble directly without ID matching.
        # Let's check if the lengths are at least the same.
        assert len(stat_y_sorted) == len(neural_y_sorted), "Target lengths differ"

    # Create ensemble lists
    oof_preds_list = [stat_oof, neural_oof]
    test_preds_list = [stat_test_preds, neural_test_preds]
    model_names = ["Statistical_LR", f"Neural_{model_name}"]

    optimizer = EnsembleOptimizer(model_names)
    weights, best_score = optimizer.optimize(oof_preds_list, neural_y_sorted)

    # Predict on test set
    final_test_preds = optimizer.predict(test_preds_list)

    # Validations
    assert np.isclose(sum(weights), 1.0), "Weights do not sum to 1"
    assert final_test_preds.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        Config.NUM_CLASSES,
    ), "Final prediction shape incorrect"

    # 7. Submission Generation
    print("\n[7] Generating Submission File...")
    submission_filename = "demo_submission.csv"
    optimizer.generate_submission(final_test_preds, neural_ids, submission_filename)

    submission_path = os.path.join(Config.SUBMISSION_DIR, submission_filename)
    assert os.path.exists(submission_path), "Submission file not created"

    # Verify file content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission head:\n{df_sub.head()}")
    assert df_sub.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        4,
    ), "Submission CSV shape incorrect (rows, id+3classes)"
    assert list(df_sub.columns) == [
        "id",
        "EAP",
        "HPL",
        "MWS",
    ], "Submission columns incorrect"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
