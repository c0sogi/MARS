import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from library.config import TRAIN_PARAMS, LEXICONS
from library.utils import set_seed, save_submission
from library.feature_engineering import Preprocessor
from library.training import train_rf, predict_rf, train_mlp, predict_mlp


def main():
    # 1. Setup
    set_seed()

    # 2. Data Processing
    # We use load_cached_data=True to leverage any pre-computed features in ./working
    preprocessor = Preprocessor()
    data = preprocessor.run(load_cached_data=True)

    # 3. Training
    # Learner A: Augmented Random Forest
    print("\nTraining Learner A: Augmented Random Forest...")
    rf_model = train_rf(data["rf_train_tab"], data["rf_train_tfidf"], data["y_train"])

    # Learner B: Dual-Branch MLP
    print("\nTraining Learner B: Dual-Branch MLP...")
    mlp_model = train_mlp(
        data["mlp_train_tab"],
        data["mlp_train_sbert"],
        data["y_train"],
        data["mlp_val_tab"],
        data["mlp_val_sbert"],
        data["y_val"],
    )

    # 4. Validation Inference & Metric
    print("\nEvaluating on Validation Set...")

    # Get predictions
    val_probs_rf = predict_rf(rf_model, data["rf_val_tab"], data["rf_val_tfidf"])
    val_probs_mlp = predict_mlp(mlp_model, data["mlp_val_tab"], data["mlp_val_sbert"])

    # Ensemble (Simple Weighted Average 0.5/0.5)
    val_probs_ensemble = 0.5 * val_probs_rf + 0.5 * val_probs_mlp

    # Calculate Metric
    val_auc = roc_auc_score(data["y_val"], val_probs_ensemble)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(data["y_val"] - val_probs_ensemble)

    # Reconstruct feature names to provide meaningful analysis
    # We need to replicate the order in Preprocessor.process_tabular
    # 1. Base numeric columns (intersection of train/test)
    df_train_meta = pd.read_csv("./metadata/train.csv")
    df_test_meta = pd.read_csv("./metadata/test.csv")
    base_cols = preprocessor.get_feature_intersection(df_train_meta, df_test_meta)

    # 2. Ratio columns
    ratio_cols = ["upvote_ratio_proxy", "comments_per_post", "raop_post_ratio"]

    # 3. Meta columns
    meta_cols = ["text_len_char", "text_len_word", "text_caps_ratio"]

    # 4. Lexicon columns
    lex_cols = [f"lexicon_density_{cat}" for cat in LEXICONS.keys()]

    feature_names = base_cols + ratio_cols + meta_cols + lex_cols

    # Verify shape matches
    if data["rf_val_tab"].shape[1] == len(feature_names):
        # Create DataFrame for correlation analysis
        df_analysis = pd.DataFrame(data["rf_val_tab"], columns=feature_names)
        df_analysis["error_magnitude"] = errors

        # Calculate correlation with error
        correlations = (
            df_analysis.corr()["error_magnitude"]
            .drop("error_magnitude")
            .abs()
            .sort_values(ascending=False)
        )

        print("Top 10 Features Correlated with Prediction Error:")
        print(correlations.head(10))
    else:
        print(
            f"Feature name mismatch (Expected {len(feature_names)}, Got {data['rf_val_tab'].shape[1]}). Skipping detailed feature correlation."
        )

    # 6. Submission Logic
    THRESHOLD = 0.6789999838498684

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Inference on Test Set
        test_probs_rf = predict_rf(rf_model, data["rf_test_tab"], data["rf_test_tfidf"])
        test_probs_mlp = predict_mlp(
            mlp_model, data["mlp_test_tab"], data["mlp_test_sbert"]
        )

        # Ensemble
        test_probs_ensemble = 0.5 * test_probs_rf + 0.5 * test_probs_mlp

        # Save
        save_submission(data["test_ids"], test_probs_ensemble)
    else:
        print(
            f"\nValidation metric {val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
