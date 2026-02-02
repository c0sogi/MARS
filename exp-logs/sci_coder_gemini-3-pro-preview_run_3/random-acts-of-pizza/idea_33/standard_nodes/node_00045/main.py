import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, suppress_warnings, Timer
from library.data_loader import load_and_clean_data
from library.feature_engineering import create_features
from library.workflow import (
    CrossValidationEngine,
    ValidationGuidedRetrainer,
    generate_submission,
)
from library.model_zoo import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    MetadataAnchor,
    StackingMetaLearner,
)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    suppress_warnings()

    print("Initializing Enhanced-Text Pent-View Stacking Ensemble...")

    # 2. Load Data
    # We use cached data if available for speed
    train_df, val_df, test_df = load_and_clean_data(load_cached_data=True)

    # Extract targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # 3. Feature Engineering
    # Generates dictionary of sparse/dense matrices for each split
    X_train, X_val, X_test = create_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Cross-Validation (Level 1)
    # Generate OOF predictions on the training set to train the Meta-Learner
    print("\n--- Running Cross-Validation ---")
    cv_engine = CrossValidationEngine()
    oof_preds, y_train_aligned = cv_engine.run_cv(X_train, y_train)

    # 5. Evaluation Phase
    # We must evaluate on the hold-out validation set using models trained ONLY on the training set.
    # We cannot use the ValidationGuidedRetrainer here because it merges Train+Val for some models.
    print("\n--- Starting Evaluation Phase ---")

    # 5a. Train Meta-Learner for Evaluation (trained on OOFs)
    meta_eval = StackingMetaLearner()
    meta_eval.fit(oof_preds, y_train_aligned)

    # 5b. Train Base Learners on Training Set Only
    base_models_eval = [
        LexicalBagger(),
        CommunityBagger(),
        SemanticBooster(),
        SemanticBagger(),
        MetadataAnchor(),
    ]

    # Helper to map model names to feature views
    def get_view(name, X_dict):
        if name == "LexicalBagger":
            return X_dict["lexical"]
        if name == "CommunityBagger":
            return X_dict["behavioral"]
        if name in ["SemanticBooster", "SemanticBagger"]:
            return X_dict["semantic"]
        if name == "MetadataAnchor":
            return X_dict["metadata"]
        raise ValueError(f"Unknown model: {name}")

    # Matrix to store Level 1 predictions for the validation set
    val_preds_matrix = np.zeros((len(y_val), len(base_models_eval)))

    with Timer("Evaluation Models Training"):
        for i, model in enumerate(base_models_eval):
            X_tr_view = get_view(model.name, X_train)
            X_val_view = get_view(model.name, X_val)

            # Fit on training data
            # Note: For SemanticBooster (XGB), we fit on full train without early stopping
            # for this strict evaluation step to avoid leaking validation info via model selection.
            model.fit(X_tr_view, y_train)

            # Predict on validation data
            val_preds_matrix[:, i] = model.predict_proba(X_val_view)

    # 5c. Generate Final Validation Predictions (Level 2)
    final_val_probs = meta_eval.predict_proba(val_preds_matrix)

    # 5d. Compute Metric
    val_auc = roc_auc_score(y_val, final_val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - final_val_probs)

    # Correlate errors with dense metadata features
    # X_val['metadata'] corresponds to Config.DENSE_FEATURES (scaled)
    meta_features = X_val["metadata"]
    feature_names = Config.DENSE_FEATURES

    correlations = []
    for idx, name in enumerate(feature_names):
        feat_vals = meta_features[:, idx]
        # Skip constant features to avoid warnings
        if np.std(feat_vals) > 1e-9:
            corr, _ = pearsonr(errors, feat_vals)
            correlations.append((name, corr))
        else:
            correlations.append((name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        print(
            f"\nMetric ({val_auc}) > Threshold ({THRESHOLD}). Proceeding to Submission..."
        )

        # Initialize Retrainer
        retrainer = ValidationGuidedRetrainer()

        # Train final models using the specific protocol:
        # - RF/Linear: Train on Train + Val
        # - XGB: Train on Train, Early Stop on Val
        # - Meta: Train on OOFs
        final_models = retrainer.train_final_models(
            X_train, y_train, X_val, y_val, oof_preds, y_train_aligned
        )

        # Generate and Save Submission
        generate_submission(final_models, X_test, test_df)

    else:
        print(
            f"\nMetric ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
