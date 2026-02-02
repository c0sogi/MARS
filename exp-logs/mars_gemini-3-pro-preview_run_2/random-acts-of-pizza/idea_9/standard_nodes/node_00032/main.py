import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.utils import set_seed
from library.preprocessor import DataPreprocessor
from library.model_trainer import EnsembleTrainer


def main():
    # 1. Setup
    set_seed(42)
    working_dir = "./working/idea_9"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    print("Starting pipeline execution...")

    # 2. Data Loading & Preprocessing
    # We use k=50 for the KNN feature as defined in the idea
    preprocessor = DataPreprocessor(k_neighbors=50)

    # Load data (this handles caching internally)
    # The dataset is small enough (~3k rows) that we don't need to subsample for a fast baseline
    print("Loading and assembling data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = (
        preprocessor.process_and_load_data(load_cached_data=False)
    )

    print(f"Data ready. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 3. Model Training
    print("Initializing trainer...")
    trainer = EnsembleTrainer(random_state=42)

    # Train and tune
    # The trainer performs grid search on C and selects the best model based on Val AUC
    print("Training and tuning ensemble...")
    trainer.tune_and_train(X_train, y_train, X_val, y_val)

    # Save the model
    model_path = os.path.join(working_dir, "ensemble_model.joblib")
    trainer.save_model(model_path)

    # 4. Validation Assessment
    # Get predictions on validation set
    y_val_pred = trainer.predict(X_val)
    val_auc = roc_auc_score(y_val, y_val_pred)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error (residuals)
    errors = np.abs(y_val - y_val_pred)

    # We want to correlate errors with features to see what drives failure.
    # X structure: [Embeddings (384) | KNN (1) | Metadata (9)]
    # We focus on KNN and Metadata for interpretability.

    # Define feature names corresponding to the last 9 columns (KNN removed)
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
    ]

    # Extract the relevant columns (last 9)
    # Embeddings are 384 dimensions. Meta starts at 384.
    X_val_analysis = X_val[:, 384:]

    if X_val_analysis.shape[1] != len(feature_names):
        print(
            f"Warning: Expected {len(feature_names)} analysis features, found {X_val_analysis.shape[1]}. Skipping detailed naming."
        )
        analysis_corrs = []
    else:
        # Compute correlations
        correlations = []
        for i, name in enumerate(feature_names):
            feat_values = X_val_analysis[:, i]
            # Handle constant features to avoid warning
            if np.std(feat_values) == 0:
                corr = 0
            else:
                corr = np.corrcoef(errors, feat_values)[0, 1]
            correlations.append((name, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top correlations between Error Magnitude and Features:")
        for name, corr in correlations:
            print(f"  {name:<50}: {corr:.4f}")

    # 6. Submission Generation
    threshold = 0.7141749705260098
    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) meets threshold ({threshold}). Generating submission..."
        )

        y_test_pred = trainer.predict(X_test)

        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": y_test_pred}
        )

        submission_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric ({val_auc}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
