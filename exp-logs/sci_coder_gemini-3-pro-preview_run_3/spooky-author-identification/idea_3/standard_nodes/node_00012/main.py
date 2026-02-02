import sys
import os
import numpy as np
import pandas as pd
import torch
import gc
import warnings
from sklearn.metrics import log_loss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library modules
from library.config import Config
from library.utils import set_seed, save_submission, calculate_log_loss
from library.data_loader import load_data, get_cv_folds
from scipy.optimize import minimize_scalar
from library.models_statistical import StatisticalExpert
from library.models_neural import TransformerExpert

# --- Monkey Patch Config for Fast Baseline ---
Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("--- Starting Runfile Execution (Simplified Ensemble) ---")
    print(f"Device: {Config.DEVICE}")
    print(f"Configuration: Epochs={Config.EPOCHS}, Model={Config.MODEL_DEBERTA}")

    # 2. Load Data
    # train_df: 80% (used for training)
    # val_df: 20% (used for weight optimization and final validation)
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=False)

    print(
        f"Train set: {train_df.shape}, Val set: {val_df.shape}, Test set: {test_df.shape}"
    )

    # Targets
    y_train = train_df["target"].values
    y_val = val_df["target"].values

    train_texts = train_df["text"].values
    val_texts = val_df["text"].values
    test_texts = test_df["text"].values

    # --- A. Statistical Expert ---
    print("\nTraining Statistical Expert...")
    model_stat = StatisticalExpert()
    model_stat.fit(train_texts, y_train)

    val_probs_stat = model_stat.predict_proba(val_texts)
    test_probs_stat = model_stat.predict_proba(test_texts)

    score_stat = calculate_log_loss(y_val, val_probs_stat)
    print(f"Statistical Expert Val LogLoss: {score_stat:.5f}")

    # --- B. Neural Expert: DeBERTa ---
    print(f"\nTraining Neural Expert ({Config.MODEL_DEBERTA})...")
    model_deberta = TransformerExpert(model_name=Config.MODEL_DEBERTA)
    # We pass val data to fit() for early stopping monitoring,
    # but we will use the final best model for prediction.
    model_deberta.fit(train_texts, y_train, val_texts, y_val)

    val_probs_deberta = model_deberta.predict_proba(val_texts)
    test_probs_deberta = model_deberta.predict_proba(test_texts)

    score_deberta = calculate_log_loss(y_val, val_probs_deberta)
    print(f"Neural Expert Val LogLoss: {score_deberta:.5f}")

    # Cleanup
    del model_deberta
    gc.collect()
    torch.cuda.empty_cache()

    # --- C. Ensemble Weight Optimization ---
    print("\nOptimizing Ensemble Weights...")

    def objective(x):
        # x is weight for statistical model
        # (1-x) is weight for neural model
        w_stat = x
        w_neural = 1 - x

        # Simple linear blend
        blended_probs = w_stat * val_probs_stat + w_neural * val_probs_deberta
        return calculate_log_loss(y_val, blended_probs)

    # Constrain weight to [0, 1]
    res = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    best_w_stat = res.x
    best_w_neural = 1 - best_w_stat

    print(f"Optimal Weights -> Stat: {best_w_stat:.4f}, Neural: {best_w_neural:.4f}")

    # Final Validation
    val_final_probs = best_w_stat * val_probs_stat + best_w_neural * val_probs_deberta
    final_metric = calculate_log_loss(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    row_indices = np.arange(len(y_val))
    true_class_probs = val_final_probs[row_indices, y_val]
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)
    error_magnitudes = -np.log(true_class_probs)

    val_meta_features = val_df["log_char_len"].values
    correlation = np.corrcoef(val_meta_features, error_magnitudes)[0, 1]
    print(f"Correlation between Error Magnitude and log_char_len: {correlation:.10f}")

    # --- Submission ---
    THRESHOLD = 0.2665362892717963
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating Submission..."
        )
        test_final_probs = (
            best_w_stat * test_probs_stat + best_w_neural * test_probs_deberta
        )
        save_submission(test_df["id"].values, test_final_probs, Config.SUBMISSION_FILE)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
