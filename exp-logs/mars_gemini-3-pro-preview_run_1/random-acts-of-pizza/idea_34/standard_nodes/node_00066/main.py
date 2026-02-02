import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, load_data, save_submission
from library.feature_engineering import FeaturePipeline
from library.random_forest import RandomForestTrainer
from library.neural_net import NeuralTrainer


def main():
    # 1. Setup
    set_seed(Config.RANDOM_STATE)
    print("Starting execution of Hybrid Ensemble Pipeline...")

    # 2. Load Data
    print("\n--- Loading Data ---")
    df_train = load_data("train", load_cached_data=True)
    df_val = load_data("val", load_cached_data=True)
    df_test = load_data("test", load_cached_data=True)

    # 3. Feature Engineering
    print("\n--- Feature Engineering ---")
    pipeline = FeaturePipeline()

    # Fit on train, transform all
    train_data = pipeline.fit_transform(df_train, "train")
    val_data = pipeline.transform(df_val, "val")
    test_data = pipeline.transform(df_test, "test")

    # 4. Train Random Forest (Stream A)
    print("\n--- Training Random Forest (Stream A) ---")
    rf_trainer = RandomForestTrainer()
    # Note: train() returns val_auc, but we will compute the ensemble AUC manually later
    rf_trainer.train(train_data, val_data)

    # Get RF Validation Predictions
    rf_val_probs = rf_trainer.predict(val_data)

    # 5. Train Neural Network (Stream B)
    print("\n--- Training Dual-Query MLP (Stream B) ---")
    # Determine metadata dimension from the processed data
    input_dim_meta = train_data["mlp_metadata"].shape[1]

    nn_trainer = NeuralTrainer(input_dim_metadata=input_dim_meta)
    nn_trainer.fit(train_data, val_data)

    # Get NN Validation Predictions
    nn_val_probs = nn_trainer.predict(val_data)

    # 6. Ensemble Evaluation
    print("\n--- Ensemble Evaluation ---")
    # Simple Weighted Average
    ensemble_val_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_val_probs) + (
        Config.ENSEMBLE_WEIGHT_MLP * nn_val_probs
    )

    val_labels = val_data["labels"]
    final_auc = roc_auc_score(val_labels, ensemble_val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_labels - ensemble_val_probs)

    # We correlate error with the metadata features (first component of features)
    # Metadata columns order based on FeaturePipeline implementation:
    # 1. requester_account_age_in_days_at_request
    # 2. requester_days_since_first_post_on_raop_at_request
    # 3. requester_number_of_comments_at_request
    # 4. requester_number_of_posts_at_request
    # 5. requester_upvotes_minus_downvotes_at_request
    # 6. Title Length
    # 7. Body Length
    # 8. Title Caps Ratio
    # 9. Body Caps Ratio
    # 10. Sentiment Neg
    # 11. Sentiment Neu
    # 12. Sentiment Pos
    # 13. Sentiment Compound

    feature_names = [
        "account_age",
        "days_since_first_post_raop",
        "num_comments",
        "num_posts",
        "upvotes_minus_downvotes",
        "title_len",
        "body_len",
        "title_caps_ratio",
        "body_caps_ratio",
        "sent_neg",
        "sent_neu",
        "sent_pos",
        "sent_compound",
    ]

    meta_matrix = val_data["mlp_metadata"]
    correlations = []

    # Calculate correlation for each feature
    for i, name in enumerate(feature_names):
        if i < meta_matrix.shape[1]:
            feat_values = meta_matrix[:, i]
            # Handle potential constant values (std=0) which cause NaN correlation
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
            else:
                corr = 0.0
            correlations.append((name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error and Metadata Features:")
    for name, corr in correlations:
        print(f"{name:<30}: {corr:.4f}")

    # 8. Submission Generation
    threshold = 0.7056961514236341
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Inference on Test Set
        rf_test_probs = rf_trainer.predict(test_data)
        nn_test_probs = nn_trainer.predict(test_data)

        # Ensemble
        ensemble_test_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_test_probs) + (
            Config.ENSEMBLE_WEIGHT_MLP * nn_test_probs
        )

        # Save
        save_submission(df_test["request_id"].values, ensemble_test_probs)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
