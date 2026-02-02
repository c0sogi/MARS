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
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import set_seed, save_submission, calculate_log_loss
from library.data_loader import load_data
from library.models_statistical import StatisticalExpert
from library.models_neural import TransformerExpert

Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("--- Starting Runfile Execution (Optimized Blending) ---")
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_DEBERTA}, Epochs: {Config.EPOCHS}")

    # 2. Load Data
    # train_df: 80% of original data
    # val_df: 20% of original data (Hold-out for blending optimization)
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=False)

    print(
        f"Train set: {train_df.shape}, Val set: {val_df.shape}, Test set: {test_df.shape}"
    )

    # 3. Train Statistical Expert (Cite Lesson 3: Weighted Voting)
    print("\n--- Training Statistical Expert ---")
    model_stat = StatisticalExpert()
    # Train on full train_df
    model_stat.fit(train_df["text"], train_df["target"])

    # Generate Predictions
    stat_val_preds = model_stat.predict_proba(val_df["text"])
    stat_test_preds = model_stat.predict_proba(test_df["text"])

    stat_score = calculate_log_loss(val_df["target"], stat_val_preds)
    print(f"Statistical Expert Val LogLoss: {stat_score:.5f}")

    # 4. Train Neural Expert (Cite Lesson 8: Orthogonal Signal, Lesson 11: Base model + More Epochs)
    print(f"\n--- Training Neural Expert ({Config.MODEL_DEBERTA}) ---")

    # Split train_df into training and internal validation for early stopping
    # We do NOT use val_df for early stopping to avoid leakage for blending optimization
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        train_df["text"],
        train_df["target"],
        test_size=0.1,
        stratify=train_df["target"],
        random_state=Config.SEED,
    )

    model_neural = TransformerExpert(model_name=Config.MODEL_DEBERTA)
    model_neural.fit(
        X_train_split.values,
        y_train_split.values,
        X_val_split.values,
        y_val_split.values,
    )

    # Generate Predictions
    neural_val_preds = model_neural.predict_proba(val_df["text"].values)
    neural_test_preds = model_neural.predict_proba(test_df["text"].values)

    neural_score = calculate_log_loss(val_df["target"], neural_val_preds)
    print(f"Neural Expert Val LogLoss: {neural_score:.5f}")

    # Cleanup
    del model_neural
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Optimize Blending Weights (Cite Lesson 11: Simple Blending)
    print("\n--- Optimizing Ensemble Weights ---")
    y_val = val_df["target"].values

    best_loss = float("inf")
    best_w = 0.5  # Weight for Statistical Model

    # Grid search for weight w (Statistical) vs (1-w) (Neural)
    # We expect Neural to be stronger, but Statistical adds orthogonality
    for w in np.linspace(0, 1, 101):
        blended = w * stat_val_preds + (1 - w) * neural_val_preds
        loss = calculate_log_loss(y_val, blended)
        if loss < best_loss:
            best_loss = loss
            best_w = w

    print(f"Best Weight for Statistical Model: {best_w:.2f}")
    print(f"Best Weight for Neural Model: {1 - best_w:.2f}")

    # 6. Final Validation
    val_final_probs = best_w * stat_val_preds + (1 - best_w) * neural_val_preds
    final_metric = calculate_log_loss(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    row_indices = np.arange(len(y_val))
    true_class_probs = val_final_probs[row_indices, y_val]

    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)
    error_magnitudes = -np.log(true_class_probs)

    val_meta_features = val_df["log_char_len"].values
    correlation = np.corrcoef(val_meta_features, error_magnitudes)[0, 1]
    print(f"Correlation between Error Magnitude and log_char_len: {correlation:.10f}")

    # 8. Submission
    THRESHOLD = 0.2665362892717963
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_final_probs = best_w * stat_test_preds + (1 - best_w) * neural_test_preds
        save_submission(test_df["id"].values, test_final_probs, Config.SUBMISSION_FILE)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
