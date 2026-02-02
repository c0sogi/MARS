import os
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Import provided libraries
from library.config import Config
from library.stacking_manager import StackingManager


def main():
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_STATE)

    print("Initializing Stacking Pipeline...")
    manager = StackingManager()

    # =========================================================================
    # 1. Feature Generation
    # =========================================================================
    print("Generating Features...")
    # Load/Compute features for all splits
    feats_train, y_train = manager.feature_engine.fit_transform(
        split="train", load_cached_data=True
    )
    feats_val, y_val = manager.feature_engine.transform(
        split="val", load_cached_data=True
    )
    feats_test, _ = manager.feature_engine.transform(
        split="test", load_cached_data=True
    )

    # =========================================================================
    # 2. Level 1 Cross-Validation (OOF Generation on Train)
    # =========================================================================
    print("Running Level 1 Cross-Validation on Train Set...")
    n_train = len(y_train)
    oof_preds = pd.DataFrame(index=range(n_train))

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
    )

    for name, model in manager.base_models.items():
        X_model = manager._get_model_input(feats_train, name)
        model_oof = np.zeros(n_train)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_model, y_train)):
            X_tr, y_tr = X_model[train_idx], y_train[train_idx]
            X_va, y_va = X_model[val_idx], y_train[val_idx]

            clf = copy.deepcopy(model)

            if name == "SemanticBooster":
                # Use fold validation set for early stopping
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                clf.fit(X_tr, y_tr)

            # Predict probabilities
            try:
                preds = clf.predict_proba(X_va)[:, 1]
            except AttributeError:
                preds = clf.predict(X_va)

            model_oof[val_idx] = preds

        oof_preds[name] = model_oof

    # =========================================================================
    # 3. Level 2 Meta-Learner Training
    # =========================================================================
    print("Training Meta-Learner on OOF Predictions...")
    manager.meta_model.fit(oof_preds, y_train)

    # =========================================================================
    # 4. Evaluation on Hold-out Validation Set
    # =========================================================================
    print("Evaluating on Hold-out Validation Set...")

    val_l1_preds = pd.DataFrame(index=range(len(y_val)))

    # Create an internal split of Train for XGBoost early stopping
    # to avoid leaking the Hold-out Val set during this evaluation phase.
    train_idx_inner, valid_idx_inner = train_test_split(
        np.arange(len(y_train)),
        test_size=0.1,
        stratify=y_train,
        random_state=Config.RANDOM_STATE,
    )

    for name, model in manager.base_models.items():
        X_train_full = manager._get_model_input(feats_train, name)
        X_val_full = manager._get_model_input(feats_val, name)

        clf = copy.deepcopy(model)

        if name == "SemanticBooster":
            # Use internal split of Train for early stopping
            X_tr_in = X_train_full[train_idx_inner]
            y_tr_in = y_train[train_idx_inner]
            X_va_in = X_train_full[valid_idx_inner]
            y_va_in = y_train[valid_idx_inner]

            clf.fit(X_tr_in, y_tr_in, eval_set=[(X_va_in, y_va_in)], verbose=False)
        else:
            clf.fit(X_train_full, y_train)

        val_l1_preds[name] = clf.predict_proba(X_val_full)[:, 1]

    # Meta-prediction on Validation Set
    val_final_preds = manager.meta_model.predict_proba(val_l1_preds)[:, 1]

    # Calculate and Print Metric
    val_auc = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {val_auc}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("Performing Failure Analysis...")
    residuals = np.abs(y_val - val_final_preds)

    # Create DataFrame for metadata features to calculate correlations
    # feats_val['metadata'] is scaled, but correlation is scale-invariant
    meta_df = pd.DataFrame(feats_val["metadata"], columns=Config.NUMERICAL_FEATURES)

    correlations = {}
    for col in meta_df.columns:
        if meta_df[col].std() > 0:
            correlations[col] = np.corrcoef(meta_df[col], residuals)[0, 1]
        else:
            correlations[col] = 0.0

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top Correlations with Prediction Error (Validation Set):")
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    threshold = 0.7085870249842536

    if val_auc > threshold:
        print(
            f"Validation AUC ({val_auc}) > Threshold ({threshold}). Generating Submission..."
        )

        # Combine Train and Val for Final Retraining
        feats_full = {}
        for key in feats_train:
            feats_full[key] = manager._vstack([feats_train[key], feats_val[key]])
        y_full = np.concatenate([y_train, y_val])

        test_l1_preds = pd.DataFrame(index=range(feats_test["metadata"].shape[0]))

        for name, model in manager.base_models.items():
            X_full = manager._get_model_input(feats_full, name)
            X_test = manager._get_model_input(feats_test, name)

            clf = copy.deepcopy(model)

            if name == "SemanticBooster":
                # For final retraining, we use the Validation set for early stopping
                # as defined in the StackingManager strategy.
                X_val_input = manager._get_model_input(feats_val, name)
                clf.fit(X_full, y_full, eval_set=[(X_val_input, y_val)], verbose=False)
            else:
                clf.fit(X_full, y_full)

            test_l1_preds[name] = clf.predict_proba(X_test)[:, 1]

        # Final Meta-Prediction
        final_test_preds = manager.meta_model.predict_proba(test_l1_preds)[:, 1]

        # Save Submission
        test_df = pd.read_parquet(Config.TEST_METADATA_PATH)
        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_test_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation AUC ({val_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
