import os
import sys
import shutil
import warnings
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.feature_engineering import FeatureExtractor
from library.dataset import get_essay_dataset, EssayDataset
from library.model_nn import DebertaRegressor
from library.trainers import Trainer
from library.postprocessing import ThresholdOptimizer

# =============================================================================
# Configuration & Setup
# =============================================================================


def setup_demo_environment():
    """
    Overrides default Config parameters to ensure the demo runs quickly
    and utilizes the provided resources efficiently for verification.
    """
    # Suppress warnings and logs for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # 1. Enable Debug Mode: Uses only 50 samples per dataset
    Config.DEBUG = True

    # 2. Reduce Training Complexity
    Config.N_FOLDS = 2  # Minimum for Cross-Validation
    Config.EPOCHS = 1  # Single pass
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUM_STEPS = 1

    # 3. Use a smaller model for demonstration speed (avoids large downloads)
    # We use a tiny BERT model to prove the pipeline works without waiting for DeBERTa-Large
    Config.MODEL_NAME = "prajjwal1/bert-tiny"
    Config.MAX_LENGTH = 128  # Reduced sequence length for speed

    # 4. Clean Working Directory to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 5. Set Seeds
    seed_everything(Config.SEED)

    print(f"Environment Setup Complete.")
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Model: {Config.MODEL_NAME}")
    print("-" * 40)


# =============================================================================
# Component Demonstrations
# =============================================================================


def demo_feature_engineering():
    print("\n[1] Demonstrating Feature Engineering...")
    fe = FeatureExtractor()

    # 1. Structural Features
    print("   Computing Structural Features...")
    # Force re-computation by setting load_cached_data=False
    s_train, s_val, s_test = fe.get_structural_features(load_cached_data=False)

    # Verification
    assert not s_train.empty, "Train structural features should not be empty"
    assert "word_count" in s_train.columns, "Missing 'word_count' feature"
    assert len(s_train) == 50, f"Expected 50 samples in Debug mode, got {len(s_train)}"
    print("   -> Structural features verified.")

    # 2. TF-IDF Features
    print("   Computing TF-IDF Features (Word)...")
    w_train, w_val, w_test = fe.get_tfidf_features(kind="word", load_cached_data=False)

    # Verification
    assert w_train.shape[0] == 50, "Train TF-IDF rows mismatch"
    assert w_train.shape[1] > 0, "TF-IDF vocabulary is empty"
    print("   -> TF-IDF features verified.")


def demo_dataset_and_model():
    print("\n[2] Demonstrating Dataset & Neural Network...")

    # 1. Dataset Loading
    print("   Loading Train Dataset...")
    ds = get_essay_dataset("train", load_cached_data=False)

    # Verification
    assert isinstance(ds, EssayDataset)
    assert len(ds) == 50
    sample = ds[0]
    assert "input_ids" in sample
    assert "labels" in sample
    print("   -> Dataset loaded and verified.")

    # 2. Model Instantiation & Forward Pass
    print("   Instantiating Model...")
    model = DebertaRegressor().to(Config.DEVICE)

    # Create a dummy batch
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    input_ids = batch["input_ids"].to(Config.DEVICE)
    mask = batch["attention_mask"].to(Config.DEVICE)

    print("   Running Forward Pass...")
    model.eval()
    with torch.no_grad():
        output = model(input_ids, mask)

    # Verification
    assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    print("   -> Model forward pass verified.")

    # Cleanup
    del model, batch, input_ids, mask
    torch.cuda.empty_cache()


def demo_full_pipeline():
    print("\n[3] Demonstrating Full Trainer Pipeline...")
    print("   Initializing Trainer...")
    trainer = Trainer()

    # 1. Semantic Branch (NN)
    # This will run the training loop for Config.N_FOLDS folds, 1 epoch each
    print("   Running Semantic Branch Training (this may take a moment)...")
    trainer.train_semantic_branch()

    # Verify outputs
    sem_oof_path = os.path.join(Config.WORKING_DIR, "train_semantic_preds.parquet")
    assert os.path.exists(sem_oof_path), "Semantic OOF predictions not found"

    # 2. Ridge Branches
    print("   Running Ridge Branch Training (Word & Char)...")
    trainer.train_ridge_branch(kind="word")
    trainer.train_ridge_branch(kind="char")

    word_oof_path = os.path.join(Config.WORKING_DIR, "train_word_preds.parquet")
    char_oof_path = os.path.join(Config.WORKING_DIR, "train_char_preds.parquet")
    assert os.path.exists(word_oof_path), "Word Ridge OOF not found"
    assert os.path.exists(char_oof_path), "Char Ridge OOF not found"

    # 3. Meta Learner
    print("   Running Meta Learner (Stacking)...")
    trainer.train_meta_learner()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    # Check Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    assert "essay_id" in sub_df.columns
    assert "score" in sub_df.columns
    assert sub_df["score"].between(1, 6).all(), "Scores must be between 1 and 6"
    print(
        f"   -> Pipeline completed successfully. Submission generated at {Config.SUBMISSION_FILE}"
    )


def demo_postprocessing():
    print("\n[4] Demonstrating Threshold Optimization...")

    # Create synthetic data
    np.random.seed(42)
    y_true = np.random.randint(1, 7, size=100)
    # Predictions are noisy versions of true labels
    y_pred_raw = y_true + np.random.normal(0, 0.5, size=100)

    optimizer = ThresholdOptimizer()

    # Fit
    print("   Fitting thresholds...")
    optimizer.fit(y_pred_raw, y_true)

    # Predict
    y_pred_discrete = optimizer.predict(y_pred_raw)

    # Verification
    assert len(y_pred_discrete) == 100
    assert np.all((y_pred_discrete >= 1) & (y_pred_discrete <= 6))

    # Check if QWK is reasonable (should be > 0 for correlated data)
    qwk = compute_qwk(y_true, y_pred_discrete)
    print(f"   -> Optimization verified. Synthetic QWK: {qwk:.4f}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    try:
        setup_demo_environment()

        # Run individual component demos
        demo_feature_engineering()
        demo_dataset_and_model()

        # Run the integrated pipeline
        demo_full_pipeline()

        # Run utility demo
        demo_postprocessing()

        print("\n" + "=" * 40)
        print("ALL DEMONSTRATIONS PASSED SUCCESSFULLY")
        print("=" * 40)

    except AssertionError as e:
        print(f"\n!!! DEMO FAILED: Assertion Error !!!\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! DEMO FAILED: Runtime Error !!!\n{e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
