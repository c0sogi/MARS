import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_manager import load_raw_data, prepare_mlm_corpus, LABEL_MAP
from library.statistical_engine import run_statistical_pipeline
from library.neural_engine import run_mlm_pretraining, train_classifier, predict_neural
from library.ensemble_optimizer import optimize_global_weights, generate_submission


def run_demo():
    print("--- Starting Demonstration Script ---")

    # 1. Configure for Speed (Debug Mode)
    # We override the static configuration to ensure the demo runs within minutes
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = (
        60  # Small subset for speed (20 samples per split approx)
    )
    Config.EPOCHS = 1  # Single epoch for fine-tuning
    Config.MLM_EPOCHS = 1  # Single epoch for MLM
    Config.BATCH_SIZE = 8  # Small batch size suitable for debug size
    Config.MLM_BATCH_SIZE = 8
    Config.TFIDF_MAX_FEATURES = 1000  # Reduce feature space for speed

    # Clean working directory to ensure we demonstrate creation/caching logic from scratch
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # 2. Load Data (Debug Subset)
    print("\n[2] Loading Data...")
    # load_raw_data respects the Config.DEBUG flag we just set
    train_df, val_df, test_df = load_raw_data(
        debug=True, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    print(f"    Train shape: {train_df.shape}")
    print(f"    Val shape:   {val_df.shape}")
    print(f"    Test shape:  {test_df.shape}")

    # Verification
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE

    # Prepare integer labels for validation evaluation later
    y_val = val_df["author"].map(LABEL_MAP).values

    # 3. Statistical Pipeline
    print("\n[3] Running Statistical Pipeline...")
    # This function handles TF-IDF extraction, model training (LR, NB), and internal blending
    # We force load_cached_data=False to demonstrate the computation
    stat_val_preds, stat_test_preds, stat_alpha = run_statistical_pipeline(
        load_cached_data=False, debug=True
    )

    print(f"    Statistical Alpha (LR Weight): {stat_alpha:.4f}")
    assert stat_val_preds.shape == (len(val_df), 3)
    assert stat_test_preds.shape == (len(test_df), 3)

    # 4. Neural Pipeline Preparation (MLM Corpus)
    print("\n[4] Preparing MLM Corpus...")
    # We use the dataframes loaded in step 2 so the corpus matches our debug subset
    mlm_texts = prepare_mlm_corpus(train_df, val_df, test_df, load_cached_data=False)
    assert len(mlm_texts) == len(train_df) + len(val_df) + len(test_df)

    # 5. Neural Branch 1: DeBERTa
    print("\n[5] Running Neural Branch 1 (DeBERTa)...")
    deberta_model_name = Config.MODEL_DEBERTA

    # A. MLM Pretraining (Domain Adaptation)
    print(f"    Pretraining {deberta_model_name}...")
    deberta_adapted_path = run_mlm_pretraining(
        deberta_model_name, mlm_texts, load_cached_data=False
    )
    assert os.path.isdir(deberta_adapted_path)

    # B. Fine-tuning
    print(f"    Fine-tuning {deberta_model_name}...")
    # Note: train_classifier expects raw labels; AuthorDataset handles mapping
    deberta_model, deberta_tokenizer, deb_val_loss = train_classifier(
        deberta_model_name,
        deberta_adapted_path,
        train_df["text"],
        train_df["author"],
        val_df["text"],
        val_df["author"],
    )

    # C. Inference
    print(f"    Inference with {deberta_model_name}...")
    deb_val_preds = predict_neural(deberta_model, deberta_tokenizer, val_df["text"])
    deb_test_preds = predict_neural(deberta_model, deberta_tokenizer, test_df["text"])

    assert deb_val_preds.shape == (len(val_df), 3)
    assert deb_test_preds.shape == (len(test_df), 3)

    # 6. Neural Branch 2: RoBERTa
    print("\n[6] Running Neural Branch 2 (RoBERTa)...")
    roberta_model_name = Config.MODEL_ROBERTA

    # A. MLM Pretraining
    print(f"    Pretraining {roberta_model_name}...")
    roberta_adapted_path = run_mlm_pretraining(
        roberta_model_name, mlm_texts, load_cached_data=False
    )

    # B. Fine-tuning
    print(f"    Fine-tuning {roberta_model_name}...")
    roberta_model, roberta_tokenizer, rob_val_loss = train_classifier(
        roberta_model_name,
        roberta_adapted_path,
        train_df["text"],
        train_df["author"],
        val_df["text"],
        val_df["author"],
    )

    # C. Inference
    print(f"    Inference with {roberta_model_name}...")
    rob_val_preds = predict_neural(roberta_model, roberta_tokenizer, val_df["text"])
    rob_test_preds = predict_neural(roberta_model, roberta_tokenizer, test_df["text"])

    # 7. Ensemble Optimization
    print("\n[7] Optimizing Ensemble...")
    # Finds the best weights to combine Statistical, DeBERTa, and RoBERTa predictions
    best_weights = optimize_global_weights(
        stat_val_preds, deb_val_preds, rob_val_preds, y_val
    )

    assert len(best_weights) == 3
    assert abs(sum(best_weights) - 1.0) < 1e-5

    # 8. Submission Generation
    print("\n[8] Generating Submission...")
    submission_df = generate_submission(
        test_df["id"], stat_test_preds, deb_test_preds, rob_test_preds, best_weights
    )

    # Final Validation
    print("\n[9] Validating Output...")
    assert os.path.exists(Config.SUBMISSION_PATH)
    assert submission_df.shape == (len(test_df), 4)  # id + 3 classes
    assert list(submission_df.columns) == ["id", "EAP", "HPL", "MWS"]

    print("\n--- Demonstration Complete Successfully ---")


if __name__ == "__main__":
    run_demo()
