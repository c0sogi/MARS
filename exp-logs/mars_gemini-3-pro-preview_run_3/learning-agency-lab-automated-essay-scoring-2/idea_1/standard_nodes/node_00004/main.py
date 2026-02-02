import os
import sys
import numpy as np
import pandas as pd
import torch
import random

# Import from the provided library files
from library.config import Config
from library.data_loader import load_essay_data
from library.feature_extractor import EmbeddingEngine
from library.model_trainer import GradientBoostingRegressor
from library.utils import compute_qwk, post_process_preds
from sklearn.preprocessing import StandardScaler


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

    X_train = engine.generate_embeddings(train_texts, "train")
    X_val = engine.generate_embeddings(val_texts, "val")
    X_test = engine.generate_embeddings(test_texts, "test")

    # Cite solution_lesson_node_00002: Explicit Injection of Meta-Features into Normalized Embedding Spaces
    print("\nInjecting advanced meta-features...")

    def get_meta_features(texts):
        features = []
        for t in texts:
            t_str = str(t)
            words = t_str.split()
            chars = len(t_str)
            word_count = len(words)

            # Sentence count (approximation)
            sentences = t_str.replace("?", ".").replace("!", ".").split(".")
            sentence_count = len([s for s in sentences if len(s.strip()) > 0])

            # Paragraph count
            paragraph_count = t_str.count("\n") + 1

            # Averages
            avg_word_len = chars / max(1, word_count)
            avg_sentence_len = word_count / max(1, sentence_count)

            # Diversity
            unique_ratio = len(set(words)) / max(1, word_count)

            features.append(
                [
                    word_count,
                    sentence_count,
                    paragraph_count,
                    avg_word_len,
                    avg_sentence_len,
                    unique_ratio,
                ]
            )

        return np.array(features)

    train_meta = get_meta_features(train_texts)
    val_meta = get_meta_features(val_texts)
    test_meta = get_meta_features(test_texts)

    scaler = StandardScaler()
    train_meta_scaled = scaler.fit_transform(train_meta)
    val_meta_scaled = scaler.transform(val_meta)
    test_meta_scaled = scaler.transform(test_meta)

    X_train = np.hstack([X_train, train_meta_scaled])
    X_val = np.hstack([X_val, val_meta_scaled])
    X_test = np.hstack([X_test, test_meta_scaled])

    # 4. Model Training
    print("\nTraining model...")
    regressor = GradientBoostingRegressor()

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
    if final_metric > 0.7479523308916254:
        print("\nGenerating submission...")
        test_preds_raw = regressor.predict(X_test)
        test_preds_int = post_process_preds(test_preds_raw)

        submission_df = pd.DataFrame({"essay_id": test_ids, "score": test_preds_int})

        # Save submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    else:
        print(f"\nMetric {final_metric} <= threshold. Skipping submission.")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
