import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import StandardScaler

# Import from the provided library files
from library.config import Config
from library.data_loader import load_essay_data
from library.feature_extractor import EmbeddingEngine
from library.model_trainer import RidgeRegressor
from library.utils import compute_qwk, post_process_preds


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(val_texts, val_scores, val_preds_raw):
    """
    Analyzes model errors on the validation set to identify systematic issues.
    """
    print("\n=== Failure Analysis ===")

    # Calculate errors
    errors = val_preds_raw - val_scores
    abs_errors = np.abs(errors)

    # Feature 1: Essay Length (Character count)
    char_lengths = np.array([len(t) for t in val_texts])

    # Feature 2: Ground Truth Score
    # (Already in val_scores)

    # Calculate correlations
    # We handle potential division by zero or constant arrays gracefully, though unlikely here
    if np.std(abs_errors) == 0:
        print("Model has zero error variance (perfect prediction or constant error).")
        return

    corr_len = np.corrcoef(char_lengths, abs_errors)[0, 1]
    corr_score = np.corrcoef(val_scores, abs_errors)[0, 1]

    print(f"Correlation between Error Magnitude and Essay Length: {corr_len:.4f}")
    print(
        f"Correlation between Error Magnitude and Ground Truth Score: {corr_score:.4f}"
    )

    if abs(corr_len) > 0.3:
        print(
            ">> Observation: Model performance varies significantly with essay length."
        )
    if abs(corr_score) > 0.3:
        print(">> Observation: Model struggles more with specific score ranges.")


def main():
    # 1. Setup
    print("Setting up environment...")
    Config.setup()
    set_seed(Config.SEED)

    # 2. Load Data
    # We load cached data if available to save time on re-runs
    print("\nLoading data...")
    train_ids, train_texts, train_scores = load_essay_data(
        "train", load_cached_data=True
    )
    val_ids, val_texts, val_scores = load_essay_data("val", load_cached_data=True)
    test_ids, test_texts, _ = load_essay_data("test", load_cached_data=True)

    # 3. Feature Extraction (Embeddings)
    # The EmbeddingEngine handles GPU usage automatically
    print("\nGenerating embeddings...")
    engine = EmbeddingEngine()

    # Generate Embeddings
    X_train_emb = engine.generate_embeddings(train_texts, "train")
    X_val_emb = engine.generate_embeddings(val_texts, "val")
    X_test_emb = engine.generate_embeddings(test_texts, "test")

    # Feature Engineering: Add Word Counts
    # EDA showed strong correlation between length and score.
    # We calculate word counts, scale them, and append to embeddings.
    print("Generating auxiliary features...")

    def get_word_counts(texts):
        return np.array([len(str(t).split()) for t in texts]).reshape(-1, 1)

    aux_train = get_word_counts(train_texts)
    aux_val = get_word_counts(val_texts)
    aux_test = get_word_counts(test_texts)

    # Scale auxiliary features
    # Embeddings are already normalized, so we only scale the counts
    scaler = StandardScaler()
    aux_train_scaled = scaler.fit_transform(aux_train)
    aux_val_scaled = scaler.transform(aux_val)
    aux_test_scaled = scaler.transform(aux_test)

    # Concatenate features
    X_train = np.hstack([X_train_emb, aux_train_scaled])
    X_val = np.hstack([X_val_emb, aux_val_scaled])
    X_test = np.hstack([X_test_emb, aux_test_scaled])

    # 4. Model Training
    print("\nTraining model...")
    regressor = RidgeRegressor()

    # We pass validation data here for internal logging, but we will compute
    # the official metric separately below to ensure compliance with requirements.
    regressor.train(X_train, train_scores, X_val, val_scores)

    # 5. Validation & Metric
    print("\nValidating...")
    # Get continuous predictions
    val_preds_raw = regressor.predict(X_val)

    # Post-process to integers [1, 6]
    val_preds_int = post_process_preds(val_preds_raw)

    # Compute QWK
    final_metric = compute_qwk(val_scores, val_preds_int)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_texts, val_scores, val_preds_raw)

    # 7. Submission
    if final_metric > 0.699053065244271:
        print("\nGenerating submission...")
        test_preds_raw = regressor.predict(X_test)
        test_preds_int = post_process_preds(test_preds_raw)

        submission_df = pd.DataFrame({"essay_id": test_ids, "score": test_preds_int})

        # Save submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    else:
        print(
            f"\nMetric ({final_metric}) did not improve baseline (0.69905). Skipping submission generation."
        )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
