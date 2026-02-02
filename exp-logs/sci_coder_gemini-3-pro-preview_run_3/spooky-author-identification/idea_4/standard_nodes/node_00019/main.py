import os
import gc
import numpy as np
import pandas as pd
import torch

# Set memory management configuration immediately
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_manager import load_raw_data, prepare_mlm_corpus, LABEL_MAP
from library.statistical_engine import run_statistical_pipeline
from library.neural_engine import run_mlm_pretraining, train_classifier, predict_neural
from library.ensemble_optimizer import optimize_global_weights, generate_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    print("=== Starting Author Identification Pipeline ===")

    # 2. Data Loading
    print("Loading data...")
    train_df, val_df, test_df = load_raw_data(debug=Config.DEBUG)

    # Prepare lists for processing
    train_texts = train_df["text"].fillna("").tolist()
    train_labels = train_df["author"].tolist()
    val_texts = val_df["text"].fillna("").tolist()
    val_labels = val_df["author"].tolist()
    test_texts = test_df["text"].fillna("").tolist()
    test_ids = test_df["id"].tolist()

    # Prepare validation targets (integers) for metric computation and optimization
    y_val_indices = val_df["author"].map(LABEL_MAP).values

    # 3. Statistical Branch Execution
    print("\n--- Phase 1: Statistical Branch ---")
    stat_val_preds, stat_test_preds, stat_alpha = run_statistical_pipeline(
        load_cached_data=True, debug=Config.DEBUG
    )
    print(f"Statistical pipeline finished. Optimal Alpha: {stat_alpha:.4f}")

    # Explicit cleanup after statistical branch
    gc.collect()

    # 4. Neural Branch Preparation
    print("\n--- Phase 2: Neural Branch ---")
    # Prepare Domain Adaptation Corpus (Train + Val + Test)
    mlm_corpus = prepare_mlm_corpus(train_df, val_df, test_df, load_cached_data=True)

    # Define models to train
    neural_models = [
        {"name": Config.MODEL_DEBERTA, "key": "deberta"},
        {"name": Config.MODEL_ROBERTA, "key": "roberta"},
    ]

    neural_results = {}

    # 5. Neural Training Loop
    for model_info in neural_models:
        # Ensure clean GPU state before each model
        torch.cuda.empty_cache()
        gc.collect()

        model_name = model_info["name"]
        key = model_info["key"]
        print(f"\nProcessing Model: {key} ({model_name})")

        # A. Domain Adaptive Pre-training (MLM)
        adapted_path = run_mlm_pretraining(
            model_name, mlm_corpus, load_cached_data=True
        )

        # B. Supervised Fine-Tuning
        # train_classifier returns the best model (loaded with best weights)
        clf_model, clf_tokenizer, best_loss = train_classifier(
            model_name, adapted_path, train_texts, train_labels, val_texts, val_labels
        )

        # C. Inference
        print(f"Generating predictions for {key}...")
        val_preds = predict_neural(clf_model, clf_tokenizer, val_texts)
        test_preds = predict_neural(clf_model, clf_tokenizer, test_texts)

        neural_results[key] = {"val": val_preds, "test": test_preds}

        # D. Cleanup to free VRAM
        del clf_model
        del clf_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # 6. Ensemble Optimization
    print("\n--- Phase 3: Ensemble Optimization ---")
    deberta_val = neural_results["deberta"]["val"]
    roberta_val = neural_results["roberta"]["val"]

    best_weights = optimize_global_weights(
        stat_val_preds, deberta_val, roberta_val, y_val_indices
    )

    # 7. Final Validation & Failure Analysis
    print("\n--- Phase 4: Validation & Analysis ---")

    # Calculate final blended probabilities
    final_val_preds = (
        best_weights[0] * stat_val_preds
        + best_weights[1] * deberta_val
        + best_weights[2] * roberta_val
    )

    # Compute and Print Final Metric
    final_metric = compute_metric(y_val_indices, final_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error magnitude and text length
    print("Performing Failure Analysis...")
    # Get probability assigned to the true class
    rows = np.arange(len(y_val_indices))
    p_true = final_val_preds[rows, y_val_indices]

    # Clip to avoid log(0)
    p_true = np.clip(p_true, 1e-15, 1 - 1e-15)

    # Error magnitude (Log Loss per sample)
    error_magnitude = -np.log(p_true)

    # Input features
    char_lengths = np.array([len(t) for t in val_texts])
    word_lengths = np.array([len(t.split()) for t in val_texts])

    # Calculate correlations
    corr_char = np.corrcoef(char_lengths, error_magnitude)[0, 1]
    corr_word = np.corrcoef(word_lengths, error_magnitude)[0, 1]

    print(f"Correlation between Error and Char Length: {corr_char:.4f}")
    print(f"Correlation between Error and Word Length: {corr_word:.4f}")

    # 8. Submission Generation
    THRESHOLD = 0.25637143429287684

    if final_metric < THRESHOLD:
        print(f"\nMetric met threshold ({THRESHOLD}). Generating submission...")
        deberta_test = neural_results["deberta"]["test"]
        roberta_test = neural_results["roberta"]["test"]

        generate_submission(
            test_ids, stat_test_preds, deberta_test, roberta_test, best_weights
        )
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")

    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
