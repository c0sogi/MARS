import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, suppress_warnings, print_metric, Timer
from library.data_processing import process_data
from library.feature_extraction import FeatureFactory
from library.training_workflow import CrossValidator, FinalRetrainer
from library.model_definitions import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    InteractionBooster,
    MetadataAnchor,
    MetaLearner,
)


def main():
    # 1. Setup
    suppress_warnings()
    set_seed(Config.RANDOM_SEED)
    Config.ensure_directories()

    print("=== Starting Pipeline Execution ===")

    # 2. Data Loading & Processing
    # We use load_cached_data=True to leverage the work done in previous steps
    train_df, val_df, test_df = process_data(load_cached_data=True)

    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # 3. Feature Extraction
    ff = FeatureFactory()

    # Load/Generate all feature views
    # Lexical
    X_train_lex, X_val_lex, X_test_lex = ff.get_lexical_features(
        train_df, val_df, test_df
    )
    # Behavioral
    X_train_beh, X_val_beh, X_test_beh = ff.get_behavioral_features(
        train_df, val_df, test_df
    )
    # Semantic
    X_train_sem, X_val_sem, X_test_sem = ff.get_semantic_features(
        train_df, val_df, test_df
    )
    # Metadata
    X_train_meta, X_val_meta, X_test_meta = ff.get_metadata_features(
        train_df, val_df, test_df
    )
    # Interaction
    X_train_int, X_val_int, X_test_int = ff.get_latent_interaction_features(
        X_train_lex,
        X_train_beh,
        X_train_meta,
        X_val_lex,
        X_val_beh,
        X_val_meta,
        X_test_lex,
        X_test_beh,
        X_test_meta,
    )

    # Organize into dictionaries for easier passing
    X_train_dict = {
        "lexical": X_train_lex,
        "behavioral": X_train_beh,
        "semantic": X_train_sem,
        "metadata": X_train_meta,
        "interaction": X_train_int,
    }
    X_val_dict = {
        "lexical": X_val_lex,
        "behavioral": X_val_beh,
        "semantic": X_val_sem,
        "metadata": X_val_meta,
        "interaction": X_val_int,
    }
    X_test_dict = {
        "lexical": X_test_lex,
        "behavioral": X_test_beh,
        "semantic": X_test_sem,
        "metadata": X_test_meta,
        "interaction": X_test_int,
    }

    # 4. Cross-Validation (Level 1 OOF)
    # This generates predictions on Train to train the Meta-Learner
    cv = CrossValidator(n_folds=Config.N_FOLDS)
    oof_preds_train = cv.run_cv(X_train_dict, y_train)

    # 5. Train Meta-Learner on OOF
    meta_learner = MetaLearner()
    meta_learner.fit(oof_preds_train, y_train)

    # 6. Validation Assessment (Hold-out Set)
    # To get a pure validation score, we train Level 1 models on Train and predict on Val.
    # For Boosters, we need an internal split of Train for early stopping to avoid leaking Val.
    print("\n=== Performing Validation Assessment ===")

    # Split Train for Early Stopping (90/10)
    # We only need indices
    train_idx, es_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.1,
        stratify=y_train,
        random_state=Config.RANDOM_SEED,
    )

    # Helper to slice
    def slice_dict(d, idx):
        return {k: v[idx] for k, v in d.items()}

    X_train_sub = slice_dict(X_train_dict, train_idx)
    y_train_sub = y_train[train_idx]

    X_es_sub = slice_dict(X_train_dict, es_idx)
    y_es_sub = y_train[es_idx]

    # Initialize Validation Prediction Matrix (n_val_samples, 6 models)
    val_preds_l1 = np.zeros((len(y_val), 6))

    with Timer("Validation Models Training"):
        # 1. Lexical Bagger
        m_lex = LexicalBagger()
        m_lex.fit(X_train_dict["lexical"], X_train_dict["metadata"], y_train)
        val_preds_l1[:, 0] = m_lex.predict_proba(
            X_val_dict["lexical"], X_val_dict["metadata"]
        )[:, 1]

        # 2. Community Bagger
        m_com = CommunityBagger()
        m_com.fit(X_train_dict["behavioral"], X_train_dict["metadata"], y_train)
        val_preds_l1[:, 1] = m_com.predict_proba(
            X_val_dict["behavioral"], X_val_dict["metadata"]
        )[:, 1]

        # 3. Semantic Booster (Use internal split for ES)
        m_sem_boost = SemanticBooster()
        m_sem_boost.fit(
            X_train_sub["semantic"],
            X_train_sub["metadata"],
            y_train_sub,
            X_semantic_val=X_es_sub["semantic"],
            X_metadata_val=X_es_sub["metadata"],
            y_val=y_es_sub,
        )
        val_preds_l1[:, 2] = m_sem_boost.predict_proba(
            X_val_dict["semantic"], X_val_dict["metadata"]
        )[:, 1]

        # 4. Semantic Bagger
        m_sem_bag = SemanticBagger()
        m_sem_bag.fit(X_train_dict["semantic"], X_train_dict["metadata"], y_train)
        val_preds_l1[:, 3] = m_sem_bag.predict_proba(
            X_val_dict["semantic"], X_val_dict["metadata"]
        )[:, 1]

        # 5. Interaction Booster (Use internal split for ES)
        m_inter = InteractionBooster()
        m_inter.fit(
            X_train_sub["interaction"],
            y_train_sub,
            X_interaction_val=X_es_sub["interaction"],
            y_val=y_es_sub,
        )
        val_preds_l1[:, 4] = m_inter.predict_proba(X_val_dict["interaction"])[:, 1]

        # 6. Metadata Anchor
        m_anch = MetadataAnchor()
        m_anch.fit(X_train_dict["metadata"], y_train)
        val_preds_l1[:, 5] = m_anch.predict_proba(X_val_dict["metadata"])[:, 1]

    # Meta-Learner Prediction on Val
    val_final_probs = meta_learner.predict_proba(val_preds_l1)[:, 1]

    # Metric Calculation
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate errors
    errors = np.abs(y_val - val_final_probs)

    # Correlate with Metadata features
    # We use the raw scaled metadata features in X_val_meta
    # Config.METADATA_DENSE_COLS lists the columns in order
    available_cols = [c for c in Config.METADATA_DENSE_COLS if c in val_df.columns]
    # Note: X_val_meta might have fewer columns if some were missing, but clean_metadata handles that.
    # We assume X_val_meta columns correspond to available_cols.

    print("Correlation of Prediction Error with Metadata Features:")
    for i, col_name in enumerate(available_cols):
        if i < X_val_meta.shape[1]:
            feat_vals = X_val_meta[:, i]
            corr, _ = pearsonr(errors, feat_vals)
            print(f"  {col_name}: {corr:.4f}")

    # 8. Submission Generation
    # Threshold check
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Use FinalRetrainer to retrain on FULL data (Train + Val) and predict on Test
        # This maximizes data usage for the competition submission
        retrainer = FinalRetrainer()

        # We pass the original OOF preds and data dictionaries
        # The retrainer handles the stacking and retraining logic internally
        test_probs = retrainer.run(
            X_train_dict, y_train, X_val_dict, y_val, X_test_dict, oof_preds_train
        )

        # Save Submission
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": test_probs,
            }
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
