import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from library.config import Config
from library.utils import get_logger, set_seed, print_metric
from library.data_loader import DataLoader
from library.model_zoo import get_base_models, get_meta_learner

logger = get_logger(__name__)


class StackingTrainer:
    def __init__(self):
        self.models = get_base_models()
        self.meta_learner = get_meta_learner()
        # Vectorizer for the Behavioral Branch (Subreddit History)
        # We use simple whitespace tokenization as subreddits are distinct tokens
        self.community_vectorizer = TfidfVectorizer(
            max_features=1000, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True
        )

    def _get_subreddit_strings(self, df):
        """Converts list of subreddits into space-separated strings."""
        return df[Config.COMMUNITY_COL].apply(
            lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
        )

    def run(self, debug_sample_size=None):
        set_seed(Config.SEED)

        # 1. Load Data
        loader = DataLoader()
        # load_cached_data=True ensures we use the cache if available, else recompute
        data = loader.load_dataset(
            load_cached_data=True, debug_sample_size=debug_sample_size
        )

        train_data = data["train"]
        val_data = data["val"]
        test_data = data["test"]
        ProfilerClass = data["CommunityProfiler"]

        y_train = train_data["y"]

        # 2. Prepare Subreddit TF-IDF for OOF (Behavioral View)
        # We fit on the training set to establish vocabulary.
        train_subs_str = self._get_subreddit_strings(train_data["metadata"])
        self.community_vectorizer.fit(train_subs_str)
        X_comm_tfidf_train = self.community_vectorizer.transform(train_subs_str)

        # 3. OOF Loop
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Storage for OOF predictions (N_samples, N_models)
        # Order: Lexical, Community, SemanticXGB, SemanticRF, Metadata
        oof_preds = np.zeros((len(y_train), 5))

        logger.info(f"Starting {Config.N_FOLDS}-Fold CV on Training Set...")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_data["metadata"], y_train)
        ):
            # --- Data Slicing ---
            # Metadata
            X_meta_train_fold = train_data["metadata"].iloc[train_idx]
            X_meta_val_fold = train_data["metadata"].iloc[val_idx]

            # Dense Feature Extraction
            dense_cols = Config.METADATA_DENSE_FEATURES
            X_dense_train_fold = X_meta_train_fold[dense_cols].values
            X_dense_val_fold = X_meta_val_fold[dense_cols].values

            # Text TF-IDF
            X_text_train_fold = train_data["tfidf"][train_idx]
            X_text_val_fold = train_data["tfidf"][val_idx]

            # Community TF-IDF
            X_comm_train_fold = X_comm_tfidf_train[train_idx]
            X_comm_val_fold = X_comm_tfidf_train[val_idx]

            # Embeddings
            X_emb_train_fold = train_data["embeddings"][train_idx]
            X_emb_val_fold = train_data["embeddings"][val_idx]

            # Targets
            y_train_fold = y_train[train_idx]

            # --- Community Profiling (Nested Target Encoding) ---
            # Fit ONLY on fold training data to prevent leakage
            profiler = ProfilerClass(vocab_size=Config.COMMUNITY_VOCAB_SIZE)
            profiler.fit(X_meta_train_fold[Config.COMMUNITY_COL], y_train_fold)

            score_train = profiler.transform(
                X_meta_train_fold[Config.COMMUNITY_COL]
            ).reshape(-1, 1)
            score_val = profiler.transform(
                X_meta_val_fold[Config.COMMUNITY_COL]
            ).reshape(-1, 1)

            # --- Feature View Assembly ---
            # 1. Lexical View (Sparse): TF-IDF Text + Dense Meta
            X_lex_train = sp.hstack([X_text_train_fold, X_dense_train_fold])
            X_lex_val = sp.hstack([X_text_val_fold, X_dense_val_fold])

            # 2. Community View (Sparse): TF-IDF History + Dense Meta
            X_com_train = sp.hstack([X_comm_train_fold, X_dense_train_fold])
            X_com_val = sp.hstack([X_comm_val_fold, X_dense_val_fold])

            # 3. Semantic View (XGB - Dense): Emb + Dense Meta + Community Score
            X_sem_xgb_train = np.hstack(
                [X_emb_train_fold, X_dense_train_fold, score_train]
            )
            X_sem_xgb_val = np.hstack([X_emb_val_fold, X_dense_val_fold, score_val])

            # 4. Semantic View (RF - Dense): Emb + Dense Meta
            X_sem_rf_train = np.hstack([X_emb_train_fold, X_dense_train_fold])
            X_sem_rf_val = np.hstack([X_emb_val_fold, X_dense_val_fold])

            # 5. Contextual View (Dense): Dense Meta Only
            X_meta_train = X_dense_train_fold
            X_meta_val = X_dense_val_fold

            # --- Train Base Models ---
            fold_models = get_base_models()

            # LexicalBagger
            fold_models["LexicalBagger"].fit(X_lex_train, y_train_fold)
            oof_preds[val_idx, 0] = fold_models["LexicalBagger"].predict_proba(
                X_lex_val
            )[:, 1]

            # CommunityBagger
            fold_models["CommunityBagger"].fit(X_com_train, y_train_fold)
            oof_preds[val_idx, 1] = fold_models["CommunityBagger"].predict_proba(
                X_com_val
            )[:, 1]

            # SemanticBooster (XGB)
            fold_models["SemanticBooster"].fit(X_sem_xgb_train, y_train_fold)
            oof_preds[val_idx, 2] = fold_models["SemanticBooster"].predict_proba(
                X_sem_xgb_val
            )[:, 1]

            # SemanticBagger
            fold_models["SemanticBagger"].fit(X_sem_rf_train, y_train_fold)
            oof_preds[val_idx, 3] = fold_models["SemanticBagger"].predict_proba(
                X_sem_rf_val
            )[:, 1]

            # MetadataAnchor
            fold_models["MetadataAnchor"].fit(X_meta_train, y_train_fold)
            oof_preds[val_idx, 4] = fold_models["MetadataAnchor"].predict_proba(
                X_meta_val
            )[:, 1]

        # 4. Train Meta Learner
        logger.info("Training Meta-Learner on OOF Predictions...")
        self.meta_learner.fit(oof_preds, y_train)

        oof_auc = roc_auc_score(
            y_train, self.meta_learner.predict_proba(oof_preds)[:, 1]
        )
        print_metric("OOF AUC", oof_auc)

        # 5. Final Retraining & Prediction
        logger.info("Performing Final Retraining...")

        # --- Prepare Full Data (Train + Val) for RF/LR ---
        full_meta = pd.concat([train_data["metadata"], val_data["metadata"]], axis=0)
        full_tfidf = sp.vstack([train_data["tfidf"], val_data["tfidf"]])
        full_emb = np.vstack([train_data["embeddings"], val_data["embeddings"]])
        full_y = np.concatenate([train_data["y"], val_data["y"]])
        dense_cols = Config.METADATA_DENSE_FEATURES

        # Re-fit Community Vectorizer on Full Data
        full_subs_str = self._get_subreddit_strings(full_meta)
        self.community_vectorizer.fit(full_subs_str)

        X_comm_full = self.community_vectorizer.transform(full_subs_str)
        X_comm_test = self.community_vectorizer.transform(
            self._get_subreddit_strings(test_data["metadata"])
        )

        # Re-fit Community Profiler on Full Data
        full_profiler = ProfilerClass(vocab_size=Config.COMMUNITY_VOCAB_SIZE)
        full_profiler.fit(full_meta[Config.COMMUNITY_COL], full_y)

        # Dense Features
        full_dense = full_meta[dense_cols].values
        test_dense = test_data["metadata"][dense_cols].values

        final_models = get_base_models()

        # --- Retrain RF/LR Models (Full Data) ---

        # Lexical
        X_lex_full = sp.hstack([full_tfidf, full_dense])
        X_lex_test = sp.hstack([test_data["tfidf"], test_dense])
        final_models["LexicalBagger"].fit(X_lex_full, full_y)
        p_lex = final_models["LexicalBagger"].predict_proba(X_lex_test)[:, 1]

        # Community
        X_com_full = sp.hstack([X_comm_full, full_dense])
        X_com_test = sp.hstack([X_comm_test, test_dense])
        final_models["CommunityBagger"].fit(X_com_full, full_y)
        p_com = final_models["CommunityBagger"].predict_proba(X_com_test)[:, 1]

        # Semantic RF
        X_sem_rf_full = np.hstack([full_emb, full_dense])
        X_sem_rf_test = np.hstack([test_data["embeddings"], test_dense])
        final_models["SemanticBagger"].fit(X_sem_rf_full, full_y)
        p_sem_rf = final_models["SemanticBagger"].predict_proba(X_sem_rf_test)[:, 1]

        # Metadata LR
        final_models["MetadataAnchor"].fit(full_dense, full_y)
        p_meta = final_models["MetadataAnchor"].predict_proba(test_dense)[:, 1]

        # --- Retrain XGBoost (Train Only + Val ES) ---
        # We use the Profiler fitted on Train Only to match the distribution
        xgb_profiler = ProfilerClass(vocab_size=Config.COMMUNITY_VOCAB_SIZE)
        xgb_profiler.fit(train_data["metadata"][Config.COMMUNITY_COL], train_data["y"])

        xgb_score_train = xgb_profiler.transform(
            train_data["metadata"][Config.COMMUNITY_COL]
        ).reshape(-1, 1)
        xgb_score_val = xgb_profiler.transform(
            val_data["metadata"][Config.COMMUNITY_COL]
        ).reshape(-1, 1)
        xgb_score_test = xgb_profiler.transform(
            test_data["metadata"][Config.COMMUNITY_COL]
        ).reshape(-1, 1)

        X_xgb_train = np.hstack(
            [
                train_data["embeddings"],
                train_data["metadata"][dense_cols].values,
                xgb_score_train,
            ]
        )
        X_xgb_val = np.hstack(
            [
                val_data["embeddings"],
                val_data["metadata"][dense_cols].values,
                xgb_score_val,
            ]
        )
        X_xgb_test = np.hstack([test_data["embeddings"], test_dense, xgb_score_test])

        final_models["SemanticBooster"].fit(
            X_xgb_train,
            train_data["y"],
            eval_set=[(X_xgb_val, val_data["y"])],
            verbose=False,
        )
        p_sem_xgb = final_models["SemanticBooster"].predict_proba(X_xgb_test)[:, 1]

        # --- Ensemble Prediction ---
        test_stack = np.column_stack([p_lex, p_com, p_sem_xgb, p_sem_rf, p_meta])
        final_preds = self.meta_learner.predict_proba(test_stack)[:, 1]

        # 6. Save Submission
        sub_df = pd.DataFrame(
            {Config.ID_COL: test_data["ids"], Config.TARGET_COL: final_preds}
        )
        sub_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
