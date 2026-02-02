import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from scipy.stats import pearsonr
import warnings

# Import library components
from library.preprocessing import load_dataset
from library.features import FeatureFactory
from library.ensemble import StackingEnsemble
from library.config import RANDOM_SEED

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # Set global seed (though config handles it mostly, good practice)
    np.random.seed(RANDOM_SEED)

    print("Starting Pipeline...")

    # ---------------------------------------------------------
    # 1. Data Loading & Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Data and Generating Features...")
    # Load raw dataframes
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    # Process features (Sparse, Dense, Embeddings, Metadata)
    ff = FeatureFactory()
    data = ff.process_data(train_df, val_df, test_df, load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Model Training (Level 1 OOF & Level 2 Meta)
    # ---------------------------------------------------------
    print("\n[Step 2] Training Stacking Ensemble (OOF + Meta-Learner)...")
    ensemble = StackingEnsemble()

    # Generate OOF predictions on Train (via 5-Fold CV)
    oof_preds = ensemble.generate_oof_predictions(data)

    # Train Meta-Learner on OOF predictions
    ensemble.train_meta_learner(oof_preds, data["y_train"])

    # ---------------------------------------------------------
    # 3. Hold-Out Validation
    # ---------------------------------------------------------
    print("\n[Step 3] Performing Validation on Hold-Out Set...")

    # We need predictions on the Validation set.
    # Since generate_oof_predictions does not persist base models, and retrain_base_models
    # trains on (Train + Val), we must manually train fresh base models on Train ONLY
    # to evaluate performance on Val correctly.

    val_base_preds = pd.DataFrame(
        index=range(len(data["y_val"])), columns=ensemble.base_models.keys()
    )

    for name, base_model in ensemble.base_models.items():
        # Clone to ensure we don't affect the main ensemble instances
        model = clone(base_model)
        view_type = ensemble.model_view_map[name]

        # Construct views
        X_train_view = ensemble._construct_feature_view(data, "train", view_type)
        X_val_view = ensemble._construct_feature_view(data, "val", view_type)
        y_train = data["y_train"]

        # Handle XGBoost specific logic
        if name == "semantic_xgb":
            # Disable early stopping for this blind validation pass to match OOF logic
            model.set_params(early_stopping_rounds=None)
            model.fit(X_train_view, y_train, verbose=False)
        else:
            model.fit(X_train_view, y_train)

        # Predict
        preds = model.predict_proba(X_val_view)[:, 1]
        val_base_preds[name] = preds

    # Generate Final Probabilities using the trained Meta-Learner
    val_final_probs = ensemble.meta_learner.predict_proba(val_base_preds)[:, 1]

    # Calculate Metric
    val_auc = roc_auc_score(data["y_val"], val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 4] Performing Failure Analysis...")
    y_val = data["y_val"]
    errors = np.abs(y_val - val_final_probs)

    # Reconstruct metadata column names for analysis
    # Logic mirrors FeatureFactory.process_metadata
    exclude_suffixes = (
        "_at_retrieval",
        "request_id",
        "requester_username",
        "source_file",
        "request_text",
        "request_title",
        "request_text_edit_aware",
        "requester_subreddits_at_request",
        "requester_received_pizza",
        "giver_username_if_known",
        "requester_user_flair",
    )

    meta_cols = []
    for col in val_df.columns:
        if col in exclude_suffixes:
            continue
        if col.endswith("_at_retrieval"):
            continue
        if not pd.api.types.is_numeric_dtype(val_df[col]):
            continue
        # Check if it exists in test (FeatureFactory logic)
        if col not in test_df.columns:
            continue
        meta_cols.append(col)

    # The metadata matrix has 'meta_cols' + 'Interaction_Feature' at the end
    X_val_meta = data["X_val_metadata"]

    if X_val_meta.shape[1] == len(meta_cols) + 1:
        correlations = []
        # Check tabular features
        for i, col_name in enumerate(meta_cols):
            feat_vals = X_val_meta[:, i]
            # Handle constant columns to avoid warnings
            if np.std(feat_vals) == 0:
                corr = 0
            else:
                corr, _ = pearsonr(errors, feat_vals)
            correlations.append((col_name, corr))

        # Check Interaction feature (last column)
        feat_vals = X_val_meta[:, -1]
        if np.std(feat_vals) == 0:
            corr_int = 0
        else:
            corr_int, _ = pearsonr(errors, feat_vals)
        correlations.append(("Interaction_Feature", corr_int))

        # Sort and Display
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print("Top 5 Features correlated with Prediction Error:")
        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")
    else:
        print(
            "Warning: Metadata column alignment failed. Skipping detailed correlation analysis."
        )

    # ---------------------------------------------------------
    # 5. Submission
    # ---------------------------------------------------------
    threshold = 0.7085870249842536

    if val_auc > threshold:
        print(
            f"\n[Step 5] Validation Metric ({val_auc}) > Threshold ({threshold}). Proceeding to Submission..."
        )

        # Retrain base models on Full Data (Train + Val)
        # This updates the instances in ensemble.base_models
        ensemble.retrain_base_models(data)

        # Generate predictions on Test set and save
        ensemble.predict(data)

    else:
        print(
            f"\n[Step 5] Validation Metric ({val_auc}) <= Threshold ({threshold}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
