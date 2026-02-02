import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
import warnings

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.features import extract_meta_features
from library.dataset import create_folds, get_test_dataset
from library.model_linear import run_expert_b
from library.model_transformer import run_expert_a
from library.meta_learner import run_meta_learner


def demo_pipeline():
    print("=== Starting Author Identification Demo Pipeline ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("[Setup] Overriding configuration for fast demonstration...")

    # Enable Debug Mode
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed

    # Reduce Training Complexity
    Config.N_FOLDS = 2  # Minimum folds for CV
    Config.EPOCHS = 1  # Single epoch
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRADIENT_ACCUMULATION_STEPS = 1

    # Use a tiny model for demonstration purposes to avoid large downloads/memory usage
    # The logic remains exactly the same as using DeBERTa-v3-Large
    Config.MODEL_NAME = "prajjwal1/bert-tiny"

    # Ensure clean working directory for the demo
    # We append _demo to paths to avoid overwriting real work if any
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_EXPERT_A_OOF = os.path.join(Config.WORKING_DIR, "oof_transformer.npy")
    Config.CACHE_EXPERT_A_TEST = os.path.join(
        Config.WORKING_DIR, "test_preds_transformer.npy"
    )
    Config.CACHE_EXPERT_B_OOF = os.path.join(Config.WORKING_DIR, "oof_linear.npy")
    Config.CACHE_EXPERT_B_TEST = os.path.join(
        Config.WORKING_DIR, "test_preds_linear.npy"
    )
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.WORKING_DIR, "checkpoints"), exist_ok=True)
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    seed_everything(Config.SEED)
    print("[Setup] Configuration updated.\n")

    # --------------------------------------------------------------------------
    # 2. Logic Verification: Metric
    # --------------------------------------------------------------------------
    print("[Validation] Verifying Log Loss Metric...")
    # Case 1: Perfect prediction
    y_true = np.array([0, 1, 2])
    y_pred_perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss_perfect = compute_log_loss(y_true, y_pred_perfect)
    # Log loss of 1.0 prob is 0.0, but clipped to eps, so essentially 0
    assert (
        loss_perfect < 1e-5
    ), f"Expected near-zero loss for perfect preds, got {loss_perfect}"

    # Case 2: Uniform prediction (random guess)
    y_pred_uniform = np.array(
        [[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]]
    )
    loss_uniform = compute_log_loss(y_true, y_pred_uniform)
    expected_loss = -np.log(1 / 3)  # approx 1.0986
    assert np.isclose(
        loss_uniform, expected_loss, atol=0.1
    ), f"Expected ~{expected_loss}, got {loss_uniform}"

    print("[Validation] Metric verification passed.\n")

    # --------------------------------------------------------------------------
    # 3. Logic Verification: Feature Extraction
    # --------------------------------------------------------------------------
    print("[Validation] Verifying Feature Extraction...")
    # Create dummy dataframe
    dummy_df = pd.DataFrame(
        {
            "text": [
                "This is a short sentence.",
                "This is a slightly longer sentence with words.",
            ]
        }
    )
    meta_feats = extract_meta_features(dummy_df)

    # Check shape and columns
    assert meta_feats.shape == (2, 3), f"Expected shape (2, 3), got {meta_feats.shape}"
    expected_cols = ["char_len", "word_count", "punct_density"]
    for col in expected_cols:
        assert col in meta_feats.columns, f"Missing column: {col}"

    # Check values
    assert meta_feats.loc[0, "word_count"] == 5, "Word count calculation incorrect"
    print("[Validation] Feature extraction verification passed.\n")

    # --------------------------------------------------------------------------
    # 4. Run Expert B (Linear Model)
    # --------------------------------------------------------------------------
    print("[Expert B] Running Linear Model Pipeline...")
    # Force reload to demonstrate training
    if os.path.exists(Config.CACHE_EXPERT_B_OOF):
        os.remove(Config.CACHE_EXPERT_B_OOF)

    oof_b, test_b = run_expert_b(load_cached_data=False, debug=True)

    # Assertions
    assert oof_b.shape[1] == 3, "Expert B OOF predictions must have 3 columns"
    assert test_b.shape[1] == 3, "Expert B Test predictions must have 3 columns"
    # In debug mode with N_FOLDS=2, OOF size matches the debug sample size (or slightly less due to split)
    # create_folds limits to DEBUG_SAMPLE_SIZE
    assert len(oof_b) <= Config.DEBUG_SAMPLE_SIZE
    print("[Expert B] Completed successfully.\n")

    # --------------------------------------------------------------------------
    # 5. Run Expert A (Transformer Model)
    # --------------------------------------------------------------------------
    print("[Expert A] Running Transformer Pipeline (Tiny Model)...")
    # Force reload
    if os.path.exists(Config.CACHE_EXPERT_A_OOF):
        os.remove(Config.CACHE_EXPERT_A_OOF)

    oof_a, test_a = run_expert_a(load_cached_data=False, debug=True)

    # Assertions
    assert oof_a.shape == oof_b.shape, "Mismatch in OOF shapes between experts"
    assert test_a.shape == test_b.shape, "Mismatch in Test shapes between experts"
    print("[Expert A] Completed successfully.\n")

    # --------------------------------------------------------------------------
    # 6. Run Meta-Learner
    # --------------------------------------------------------------------------
    print("[Meta-Learner] Running Stacking Ensemble...")

    run_meta_learner(
        expert_a_oof=oof_a,
        expert_a_test=test_a,
        expert_b_oof=oof_b,
        expert_b_test=test_b,
        debug=True,
    )

    # --------------------------------------------------------------------------
    # 7. Final Submission Validation
    # --------------------------------------------------------------------------
    print("\n[Final Check] Validating Submission File...")

    if not os.path.exists(Config.SUBMISSION_FILE_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_FILE_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)

    # Check columns
    expected_sub_cols = ["id", "EAP", "HPL", "MWS"]
    assert (
        list(sub_df.columns) == expected_sub_cols
    ), f"Invalid submission columns: {sub_df.columns}"

    # Check row count (should match test_a length)
    assert len(sub_df) == len(
        test_a
    ), f"Submission row count {len(sub_df)} does not match test predictions {len(test_a)}"

    # Check probabilities sum roughly to 1 (rescaling happens in scoring, but raw output usually sums to 1)
    row_sums = sub_df[["EAP", "HPL", "MWS"]].sum(axis=1)
    # Allow small floating point error
    assert np.allclose(
        row_sums, 1.0, atol=1e-5
    ), "Submission probabilities do not sum to 1"

    print(f"[Final Check] Submission file valid. Shape: {sub_df.shape}")
    print(f"[Final Check] File saved to: {Config.SUBMISSION_FILE_PATH}")
    print("\n=== Demo Pipeline Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
