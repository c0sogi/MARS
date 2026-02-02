import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

# Import provided libraries
from library.utils import (
    set_seed,
    score_predictions,
    create_submission,
    clip_probabilities,
)
from library.data_factory import DataFactory
from library.pipeline_definitions import get_expert_library
from library.ensemble_strategy import GreedySelector

# Constants
RANDOM_SEED = 42
SUBMISSION_PATH = "./submission/submission.csv"


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    print("Initializing MS-DIPGE Workflow...")

    # 2. Data Loading
    print("Loading datasets...")
    factory = DataFactory()
    # Load cached data if available to save time; DataFactory handles morphometric extraction
    df_train_full, df_val_full, df_test_full = factory.load_datasets(
        load_cached_data=True
    )

    # Extract Feature Groups
    feature_groups = factory.get_feature_groups(df_train_full)

    # Prepare X and y
    target_col = "species"

    # Encode labels
    # Fit LabelEncoder on all known species (train + val) to ensure consistency
    le = LabelEncoder()
    all_species = pd.concat(
        [df_train_full[target_col], df_val_full[target_col]]
    ).unique()
    le.fit(all_species)

    y_train = le.transform(df_train_full[target_col])
    y_val = le.transform(df_val_full[target_col])

    # Filter feature columns
    feature_cols = feature_groups["all_features"]

    X_train = df_train_full[feature_cols].copy()
    X_val = df_val_full[feature_cols].copy()
    X_test = df_test_full[feature_cols].copy()

    test_ids = df_test_full["id"].values
    class_names = list(le.classes_)

    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )
    print(f"Number of classes: {len(class_names)}")

    # 3. Define Expert Library
    print("Initializing Expert Library...")
    # Using shrinkage levels [0.001, 0.01] as per the MS-DIPGE strategy
    experts = get_expert_library(feature_groups, shrinkage_levels=[0.001, 0.01])
    print(f"Defined {len(experts)} experts.")

    # 4. Phase 1: Training Experts & Selection
    print("\n--- Phase 1: Training Experts & Selection ---")
    val_predictions = {}

    # Train each expert on Training set and predict on Validation set
    for name, pipeline in experts.items():
        # print(f"Training {name}...", end="\r")
        try:
            pipeline.fit(X_train, y_train)
            # Predict probabilities
            preds = pipeline.predict_proba(X_val)
            val_predictions[name] = preds
        except Exception as e:
            print(f"\nFailed to train {name}: {e}")
    print("\nTraining complete.")

    # Run Greedy Selection
    print("Running Greedy Forward Selection...")
    selector = GreedySelector(max_iterations=20, tolerance=1e-6, verbose=True)
    selector.fit(val_predictions, y_val, labels=None)

    best_val_score = selector.best_score_
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {best_val_score:.15f}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Get ensemble predictions on validation
    ensemble_val_probs = selector.predict(val_predictions)

    # Calculate per-sample log loss approximation (negative log likelihood of true class)
    ensemble_val_probs_clipped = clip_probabilities(ensemble_val_probs)

    # Extract prob of true class using integer indexing
    rows = np.arange(len(y_val))
    true_class_probs = ensemble_val_probs_clipped[rows, y_val]
    sample_losses = -np.log(true_class_probs)

    # Create aggregate features for analysis
    analysis_df = X_val.copy()
    analysis_df["mean_margin"] = analysis_df[feature_groups["margin"]].mean(axis=1)
    analysis_df["mean_shape"] = analysis_df[feature_groups["shape"]].mean(axis=1)
    analysis_df["mean_texture"] = analysis_df[feature_groups["texture"]].mean(axis=1)

    # Check correlations for morph features + aggregates
    check_cols = feature_groups["morph"] + ["mean_margin", "mean_shape", "mean_texture"]
    correlations = []

    print("Correlation between Error (LogLoss) and Features:")
    for col in check_cols:
        if col in analysis_df.columns:
            vals = analysis_df[col].values
            # Avoid constant columns
            if np.std(vals) > 1e-9:
                corr, _ = pearsonr(sample_losses, vals)
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Phase 2: Retraining Selected Experts on Full Data
    # We proceed with submission generation if the score is reasonable.
    if best_val_score < 10.0:
        print("\n--- Phase 2: Retraining Selected Experts on Full Data ---")

        # Combine Train and Val
        X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Identify selected experts
        selected_expert_names = list(selector.weights_.keys())
        test_predictions_dict = {}

        for name in selected_expert_names:
            # print(f"Retraining {name}...", end="\r")
            # Retrieve the pipeline definition again (fresh instance via dictionary access)
            # and refit on the full dataset.
            pipeline = experts[name]
            pipeline.fit(X_full, y_full)

            # Predict on Test
            test_preds = pipeline.predict_proba(X_test)
            test_predictions_dict[name] = test_preds

        print("\nRetraining complete.")

        # Weighted Average for Submission using weights learned in Phase 1
        print("Generating Ensemble Predictions for Test Set...")
        final_test_probs = selector.predict(test_predictions_dict)

        # Save Submission
        create_submission(
            test_ids, final_test_probs, class_names, output_path=SUBMISSION_PATH
        )
    else:
        print(f"Validation score {best_val_score} is too high. Skipping submission.")


if __name__ == "__main__":
    main()
