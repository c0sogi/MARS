import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Import from provided libraries
from library.config import Config
from library.utils import get_logger, set_seed
from library.data_loader import DataLoader
from library.model_zoo import get_base_models, get_meta_learner

logger = get_logger(__name__)


def get_subreddit_strings(df):
    """Converts list of subreddits into space-separated strings."""
    return df[Config.COMMUNITY_COL].apply(
        lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
    )


def main():
    set_seed(Config.SEED)

    # 1. Load Data
    logger.info("Loading data...")
    loader = DataLoader()
    # Use load_cached_data=True to utilize pre-computed features
    data = loader.load_dataset(load_cached_data=True)

    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]
    ProfilerClass = data["CommunityProfiler"]

    y_train = train_data["y"]
    y_val = val_data["y"]

    # 2. Prepare Subreddit TF-IDF (Behavioral View)
    # We fit on the training set to establish vocabulary.
    community_vectorizer = TfidfVectorizer(
        max_features=1000, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True
    )

    train_subs_str = get_subreddit_strings(train_data["metadata"])
    val_subs_str = get_subreddit_strings(val_data["metadata"])
    test_subs_str = get_subreddit_strings(test_data["metadata"])

    community_vectorizer.fit(train_subs_str)

    X_comm_tfidf_train = community_vectorizer.transform(train_subs_str)
    X_comm_tfidf_val = community_vectorizer.transform(val_subs_str)
    X_comm_tfidf_test = community_vectorizer.transform(test_subs_str)

    # 3. OOF Loop for Meta-Learner Training
    # We need to generate OOF predictions on the Training set to train the Level 2 model.
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # 5 base models
    oof_preds = np.zeros((len(y_train), 5))

    logger.info(
        f"Starting {Config.N_FOLDS}-Fold CV on Training Set for OOF generation..."
    )

    for fold, (train_idx, cv_val_idx) in enumerate(
        skf.split(train_data["metadata"], y_train)
    ):
        # --- Data Slicing ---
        # Metadata
        X_meta_tr = train_data["metadata"].iloc[train_idx]
        X_meta_cv = train_data["metadata"].iloc[cv_val_idx]

        # Dense Features
        dense_cols = Config.METADATA_DENSE_FEATURES
        X_dense_tr = X_meta_tr[dense_cols].values
        X_dense_cv = X_meta_cv[dense_cols].values

        # Text TF-IDF
        X_text_tr = train_data["tfidf"][train_idx]
        X_text_cv = train_data["tfidf"][cv_val_idx]

        # Community TF-IDF
        X_comm_tr = X_comm_tfidf_train[train_idx]
        X_comm_cv = X_comm_tfidf_train[cv_val_idx]

        # Embeddings
        X_emb_tr = train_data["embeddings"][train_idx]
        X_emb_cv = train_data["embeddings"][cv_val_idx]

        # Targets
        y_tr = y_train[train_idx]
        y_cv = y_train[cv_val_idx]

        # --- Community Profiling (Nested Target Encoding) ---
        # Fit ONLY on fold training data to prevent leakage
        profiler = ProfilerClass(vocab_size=Config.COMMUNITY_VOCAB_SIZE)
        profiler.fit(X_meta_tr[Config.COMMUNITY_COL], y_tr)

        score_tr = profiler.transform(X_meta_tr[Config.COMMUNITY_COL]).reshape(-1, 1)
        score_cv = profiler.transform(X_meta_cv[Config.COMMUNITY_COL]).reshape(-1, 1)

        # --- Assemble Views ---
        # 1. Lexical View (Sparse): TF-IDF Text + Dense Meta
        X_lex_tr = sp.hstack([X_text_tr, X_dense_tr])
        X_lex_cv = sp.hstack([X_text_cv, X_dense_cv])

        # 2. Community View (Sparse): TF-IDF History + Dense Meta
        X_com_tr = sp.hstack([X_comm_tr, X_dense_tr])
        X_com_cv = sp.hstack([X_comm_cv, X_dense_cv])

        # 3. Semantic View (XGB - Dense): Emb + Dense Meta + Community Score
        X_sem_xgb_tr = np.hstack([X_emb_tr, X_dense_tr, score_tr])
        X_sem_xgb_cv = np.hstack([X_emb_cv, X_dense_cv, score_cv])

        # 4. Semantic View (RF - Dense): Emb + Dense Meta
        X_sem_rf_tr = np.hstack([X_emb_tr, X_dense_tr])
        X_sem_rf_cv = np.hstack([X_emb_cv, X_dense_cv])

        # 5. Contextual View (Dense): Dense Meta Only
        X_meta_tr_view = X_dense_tr
        X_meta_cv_view = X_dense_cv

        # --- Train Base Models ---
        fold_models = get_base_models()

        # LexicalBagger
        fold_models["LexicalBagger"].fit(X_lex_tr, y_tr)
        oof_preds[cv_val_idx, 0] = fold_models["LexicalBagger"].predict_proba(X_lex_cv)[
            :, 1
        ]

        # CommunityBagger
        fold_models["CommunityBagger"].fit(X_com_tr, y_tr)
        oof_preds[cv_val_idx, 1] = fold_models["CommunityBagger"].predict_proba(
            X_com_cv
        )[:, 1]

        # SemanticBooster (XGB)
        fold_models["SemanticBooster"].fit(
            X_sem_xgb_tr, y_tr, eval_set=[(X_sem_xgb_cv, y_cv)], verbose=False
        )
        oof_preds[cv_val_idx, 2] = fold_models["SemanticBooster"].predict_proba(
            X_sem_xgb_cv
        )[:, 1]

        # SemanticBagger
        fold_models["SemanticBagger"].fit(X_sem_rf_tr, y_tr)
        oof_preds[cv_val_idx, 3] = fold_models["SemanticBagger"].predict_proba(
            X_sem_rf_cv
        )[:, 1]

        # MetadataAnchor
        fold_models["MetadataAnchor"].fit(X_meta_tr_view, y_tr)
        oof_preds[cv_val_idx, 4] = fold_models["MetadataAnchor"].predict_proba(
            X_meta_cv_view
        )[:, 1]

    # 4. Train Meta-Learner
    logger.info("Training Meta-Learner...")
    meta_learner = get_meta_learner()
    meta_learner.fit(oof_preds, y_train)

    # 5. Train Base Models on Full Training Set (for Evaluation on Hold-out Val)
    logger.info("Training Base Models on Full Train Set for Validation...")

    # Profiler on full train
    profiler_full = ProfilerClass(vocab_size=Config.COMMUNITY_VOCAB_SIZE)
    profiler_full.fit(train_data["metadata"][Config.COMMUNITY_COL], y_train)

    score_train_full = profiler_full.transform(
        train_data["metadata"][Config.COMMUNITY_COL]
    ).reshape(-1, 1)
    score_val_full = profiler_full.transform(
        val_data["metadata"][Config.COMMUNITY_COL]
    ).reshape(-1, 1)

    dense_cols = Config.METADATA_DENSE_FEATURES
    X_dense_train = train_data["metadata"][dense_cols].values
    X_dense_val = val_data["metadata"][dense_cols].values

    # Assemble Full Train Views
    X_lex_train = sp.hstack([train_data["tfidf"], X_dense_train])
    X_com_train = sp.hstack([X_comm_tfidf_train, X_dense_train])
    X_sem_xgb_train = np.hstack(
        [train_data["embeddings"], X_dense_train, score_train_full]
    )
    X_sem_rf_train = np.hstack([train_data["embeddings"], X_dense_train])
    X_meta_train_view = X_dense_train

    # Assemble Val Views
    X_lex_val = sp.hstack([val_data["tfidf"], X_dense_val])
    X_com_val = sp.hstack([X_comm_tfidf_val, X_dense_val])
    X_sem_xgb_val = np.hstack([val_data["embeddings"], X_dense_val, score_val_full])
    X_sem_rf_val = np.hstack([val_data["embeddings"], X_dense_val])
    X_meta_val_view = X_dense_val

    # Fit Base Models
    base_models = get_base_models()

    base_models["LexicalBagger"].fit(X_lex_train, y_train)
    p_lex_val = base_models["LexicalBagger"].predict_proba(X_lex_val)[:, 1]

    base_models["CommunityBagger"].fit(X_com_train, y_train)
    p_com_val = base_models["CommunityBagger"].predict_proba(X_com_val)[:, 1]

    # For XGB validation eval, split train for ES
    es_tr_idx, es_val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.1,
        stratify=y_train,
        random_state=Config.SEED,
    )

    base_models["SemanticBooster"].fit(
        X_sem_xgb_train[es_tr_idx],
        y_train[es_tr_idx],
        eval_set=[(X_sem_xgb_train[es_val_idx], y_train[es_val_idx])],
        verbose=False,
    )
    p_sem_xgb_val = base_models["SemanticBooster"].predict_proba(X_sem_xgb_val)[:, 1]

    base_models["SemanticBagger"].fit(X_sem_rf_train, y_train)
    p_sem_rf_val = base_models["SemanticBagger"].predict_proba(X_sem_rf_val)[:, 1]

    base_models["MetadataAnchor"].fit(X_meta_train_view, y_train)
    p_meta_val = base_models["MetadataAnchor"].predict_proba(X_meta_val_view)[:, 1]

    # 6. Predict on Val using Meta-Learner
    val_stack = np.column_stack(
        [p_lex_val, p_com_val, p_sem_xgb_val, p_sem_rf_val, p_meta_val]
    )
    val_preds = meta_learner.predict_proba(val_stack)[:, 1]

    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_val - val_preds)

    # Correlate errors with dense metadata
    val_meta_df = val_data["metadata"].copy()
    val_meta_df["error"] = errors

    # Select numerical columns
    num_cols = val_meta_df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = {}
    for col in num_cols:
        if col != "error":
            # Handle potential NaNs if any (though imputation should have handled it)
            if val_meta_df[col].std() > 0:
                corr = val_meta_df[col].corr(val_meta_df["error"])
                correlations[col] = corr

    # Sort and print top 5
    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]) if pd.notnull(x[1]) else 0,
        reverse=True,
    )
    print("Top 5 Feature Correlations with Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    # 8. Submission
    threshold = 0.7138293787137718
    if val_auc > threshold:
        logger.info("Validation metric met. Generating submission...")

        # Retrain on Train + Val for maximum performance
        logger.info("Retraining on Train + Val...")

        # Combine Data
        full_meta = pd.concat([train_data["metadata"], val_data["metadata"]], axis=0)
        full_tfidf = sp.vstack([train_data["tfidf"], val_data["tfidf"]])
        full_emb = np.vstack([train_data["embeddings"], val_data["embeddings"]])
        full_y = np.concatenate([y_train, y_val])

        # Re-fit Community Vectorizer
        full_subs_str = get_subreddit_strings(full_meta)
        community_vectorizer.fit(full_subs_str)
        X_comm_tfidf_full = community_vectorizer.transform(full_subs_str)
        X_comm_tfidf_test = community_vectorizer.transform(test_subs_str)

        # Re-fit Profiler
        profiler_final = ProfilerClass(vocab_size=Config.COMMUNITY_VOCAB_SIZE)
        profiler_final.fit(full_meta[Config.COMMUNITY_COL], full_y)

        score_full = profiler_final.transform(full_meta[Config.COMMUNITY_COL]).reshape(
            -1, 1
        )
        score_test = profiler_final.transform(
            test_data["metadata"][Config.COMMUNITY_COL]
        ).reshape(-1, 1)

        # Dense
        X_dense_full = full_meta[dense_cols].values
        X_dense_test = test_data["metadata"][dense_cols].values

        # Assemble Views
        X_lex_full = sp.hstack([full_tfidf, X_dense_full])
        X_lex_test = sp.hstack([test_data["tfidf"], X_dense_test])

        X_com_full = sp.hstack([X_comm_tfidf_full, X_dense_full])
        X_com_test = sp.hstack([X_comm_tfidf_test, X_dense_test])

        X_sem_xgb_full = np.hstack([full_emb, X_dense_full, score_full])
        X_sem_xgb_test = np.hstack([test_data["embeddings"], X_dense_test, score_test])

        X_sem_rf_full = np.hstack([full_emb, X_dense_full])
        X_sem_rf_test = np.hstack([test_data["embeddings"], X_dense_test])

        X_meta_full_view = X_dense_full
        X_meta_test_view = X_dense_test

        # Fit Base Models
        final_models = get_base_models()

        final_models["LexicalBagger"].fit(X_lex_full, full_y)
        p_lex_test = final_models["LexicalBagger"].predict_proba(X_lex_test)[:, 1]

        final_models["CommunityBagger"].fit(X_com_full, full_y)
        p_com_test = final_models["CommunityBagger"].predict_proba(X_com_test)[:, 1]

        # For XGB, split full data for ES
        es_tr_idx, es_val_idx = train_test_split(
            np.arange(len(full_y)),
            test_size=0.1,
            stratify=full_y,
            random_state=Config.SEED,
        )
        final_models["SemanticBooster"].fit(
            X_sem_xgb_full[es_tr_idx],
            full_y[es_tr_idx],
            eval_set=[(X_sem_xgb_full[es_val_idx], full_y[es_val_idx])],
            verbose=False,
        )
        p_sem_xgb_test = final_models["SemanticBooster"].predict_proba(X_sem_xgb_test)[
            :, 1
        ]

        final_models["SemanticBagger"].fit(X_sem_rf_full, full_y)
        p_sem_rf_test = final_models["SemanticBagger"].predict_proba(X_sem_rf_test)[
            :, 1
        ]

        final_models["MetadataAnchor"].fit(X_meta_full_view, full_y)
        p_meta_test = final_models["MetadataAnchor"].predict_proba(X_meta_test_view)[
            :, 1
        ]

        # Predict Test
        test_stack = np.column_stack(
            [p_lex_test, p_com_test, p_sem_xgb_test, p_sem_rf_test, p_meta_test]
        )
        test_preds = meta_learner.predict_proba(test_stack)[:, 1]

        # Save
        sub_df = pd.DataFrame(
            {Config.ID_COL: test_data["ids"], Config.TARGET_COL: test_preds}
        )
        sub_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    else:
        logger.warning("Validation metric threshold not met. Skipping submission.")


if __name__ == "__main__":
    main()
