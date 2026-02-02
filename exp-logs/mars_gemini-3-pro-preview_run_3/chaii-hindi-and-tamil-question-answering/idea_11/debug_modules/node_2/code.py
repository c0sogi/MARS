import os
import sys
import pandas as pd
import torch
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_jaccard_score
from library.tapt_engine import run_tapt_pretraining
from library.qa_engine import QAEngine
from library.inference_engine import InferenceEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("=== Starting Demonstration of QA Pipeline ===")

    # ----------------------------------------------------------------
    # 1. Configuration Setup
    # ----------------------------------------------------------------
    # We override the default configuration to ensure the demo runs quickly.
    print("\n[1] Configuring environment...")

    # Use a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run_v1"

    # Update paths dependent on WORKING_DIR
    Config.TAPT_CACHE_DIR = os.path.join(Config.WORKING_DIR, "tapt_cache")
    Config.TAPT_MODEL_DIR = os.path.join(Config.WORKING_DIR, "tapt_model_finetuned")
    Config.QA_CACHE_DIR = os.path.join(Config.WORKING_DIR, "qa_cache")
    Config.QA_MODEL_DIR = os.path.join(Config.WORKING_DIR, "qa_models")

    # Set a custom submission file path for this demo
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Reduce training parameters for speed optimization
    Config.EPOCHS = 1  # Run only 1 epoch for QA
    Config.TAPT_EPOCHS = 1  # Run only 1 epoch for TAPT
    Config.N_FOLDS = 2  # Run 2 folds to demonstrate the CV loop
    Config.DEBUG = True  # Truncate TAPT data to 50 examples

    # Initialize directories based on new config
    Config.setup()
    Config.print_config()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ----------------------------------------------------------------
    # 2. Task-Adaptive Pretraining (TAPT)
    # ----------------------------------------------------------------
    print("\n[2] Running TAPT (Task-Adaptive Pretraining)...")
    # This fine-tunes the base model on the domain text (unsupervised).
    # With DEBUG=True, this uses only 50 examples for demonstration.
    run_tapt_pretraining(load_cached_data=False)

    # Verify TAPT output
    # The trainer saves config.json and model weights
    expected_tapt_config = os.path.join(Config.TAPT_MODEL_DIR, "config.json")
    if not os.path.exists(expected_tapt_config):
        raise FileNotFoundError(f"TAPT model not saved at {Config.TAPT_MODEL_DIR}")
    print("TAPT completed and model artifacts verified.")

    # ----------------------------------------------------------------
    # 3. Question Answering Training (K-Fold)
    # ----------------------------------------------------------------
    print("\n[3] Running QA K-Fold Training...")
    qa_engine = QAEngine()

    # This trains the model on the labeled dataset using the TAPT model as a base.
    # It runs for Config.N_FOLDS (2) folds, 1 epoch each.
    qa_engine.run_k_fold_training()

    # Verify QA models were saved
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.QA_MODEL_DIR, f"model_fold_{fold}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"QA model for fold {fold} not found at {model_path}"
            )
    print(f"QA Training completed. {Config.N_FOLDS} models saved successfully.")

    # ----------------------------------------------------------------
    # 4. Inference and Submission
    # ----------------------------------------------------------------
    print("\n[4] Running Ensemble Inference...")
    inference_engine = InferenceEngine()

    # This generates predictions on the test set using majority voting from the trained folds.
    inference_engine.ensemble_predict()

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE, keep_default_na=False)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Check for required columns
    required_cols = {"id", "PredictionString"}
    if not required_cols.issubset(df_sub.columns):
        raise ValueError(f"Submission missing columns. Found: {df_sub.columns}")

    # Check for non-empty predictions (sanity check)
    non_empty_count = df_sub["PredictionString"].str.len().gt(0).sum()
    print(f"Non-empty predictions: {non_empty_count} / {len(df_sub)}")

    # ----------------------------------------------------------------
    # 5. Metric Demonstration
    # ----------------------------------------------------------------
    print("\n[5] Demonstrating Jaccard Metric...")
    # Example usage of the provided metric function to verify logic
    ground_truth = ["apple pie", "machine learning"]
    predictions = ["apple tart", "machine learning"]

    score = compute_jaccard_score(ground_truth, predictions)
    print(f"Ground Truth: {ground_truth}")
    print(f"Predictions:  {predictions}")
    print(f"Computed Jaccard Score: {score:.4f}")

    # Verify logic:
    # 1. "apple pie" vs "apple tart" -> intersection {"apple"}, union {"apple", "pie", "tart"} -> 1/3
    # 2. "machine learning" vs "machine learning" -> 1.0
    # Avg = (0.3333 + 1.0) / 2 = 0.6667
    expected_score = (1.0 / 3.0 + 1.0) / 2.0
    assert abs(score - expected_score) < 1e-6, "Jaccard calculation mismatch"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
