import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_loader import load_and_preprocess_data
from library.feature_engine import FeaturePipeline
from library.model_stack import QuadStackingClassifier


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Initialization
    set_seed(Config.RANDOM_SEED)

    # 2. Load Data
    # Using cached data as requested to optimize runtime
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=True)

    # 3. Feature Engineering
    pipeline = FeaturePipeline()
    # Generate feature views (Lexical, Behavioral, Semantic, Contextual)
    # Using cached features if available to speed up the process
    train_feats, val_feats, test_feats = pipeline.create_views(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Extract targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # 4. Model Training
    print("\nInitializing and training QuadStackingClassifier...")
    model = QuadStackingClassifier()
    # Fit on training data (internal CV handles meta-learner training)
    model.fit(train_feats, y_train)

    # 5. Validation
    print("\nEvaluating on Validation set...")
    val_probs = model.predict_proba(val_feats)
    val_auc = roc_auc_score(y_val, val_probs)

    # Print exact metric format required
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nFailure Analysis (Correlation of Error with Contextual Features):")
    errors = np.abs(y_val - val_probs)

    # Feature names corresponding to the dense contextual vector
    # Order must match FeaturePipeline._extract_global_metadata
    feature_names = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "request_hour",
        "request_dow",
        "text_len_char",
        "text_len_word",
    ]

    X_val_ctx = val_feats["contextual"]
    correlations = []

    for i, name in enumerate(feature_names):
        if i < X_val_ctx.shape[1]:
            feat_vals = X_val_ctx[:, i]
            # Avoid warning if feature is constant
            if np.std(feat_vals) > 1e-9:
                corr, _ = pearsonr(feat_vals, errors)
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    threshold = 0.6913548345419015
    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")
        test_probs = model.predict_proba(test_feats)
        test_ids = test_df[Config.ID_COL].values

        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: test_probs}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {val_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
