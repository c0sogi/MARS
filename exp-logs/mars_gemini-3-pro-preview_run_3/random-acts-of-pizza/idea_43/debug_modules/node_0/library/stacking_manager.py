import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.data_loader import load_datasets, get_text_data, get_metadata, get_target
from library.feature_generators import (
    TextProcessor,
    SentenceEmbedder,
    LatentCommunityInjector,
    MetadataAugmenter,
)
from library.model_definitions import get_base_learner, get_meta_learner


class HexStackEnsemble:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.sentence_embedder = SentenceEmbedder()

        # We need separate instances of transformers for the final retraining phase
        self.final_nmf_injector = LatentCommunityInjector()
        self.final_meta_augmenter = MetadataAugmenter()
        self.final_text_tfidf = None
        self.final_community_tfidf = None
        self.final_scaler = StandardScaler()

        # Models storage
        self.base_learners = {}
        self.meta_learner = None

        # Feature storage for final prediction
        self.final_transformers_fitted = False

    def _get_sparse_text_features(self, df, vectorizer=None, fit=False):
        """Generates TF-IDF features for the Lexical Branch."""
        text_series = self.text_processor.process(df)

        if fit:
            # "Sparse Features: TF-IDF (Unigrams/Bigrams) with sublinear_tf=True and min_df=5"
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                sublinear_tf=True,
                max_features=10000,  # Reasonable limit for sparse bagger
                stop_words="english",
            )
            features = vectorizer.fit_transform(text_series)
            return features, vectorizer
        else:
            if vectorizer is None:
                raise ValueError("Vectorizer must be provided if fit=False")
            features = vectorizer.transform(text_series)
            return features

    def _get_sparse_community_features(self, df, vectorizer=None, fit=False):
        """Generates TF-IDF features for the Community Branch."""
        # Extract subreddit string
        if "requester_subreddits_at_request" in df.columns:
            sub_series = df["requester_subreddits_at_request"].apply(
                lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
            )
        else:
            sub_series = pd.Series(["" for _ in range(len(df))])

        if fit:
            # "strictly limit the vocabulary to the Top 1,000 subreddits"
            vectorizer = TfidfVectorizer(
                max_features=Config.MAX_SUBREDDITS_VOCAB,
                token_pattern=r"(?u)\b\w+\b",
                stop_words="english",
            )
            features = vectorizer.fit_transform(sub_series)
            return features, vectorizer
        else:
            if vectorizer is None:
                raise ValueError("Vectorizer must be provided if fit=False")
            features = vectorizer.transform(sub_series)
            return features

    def _get_dense_semantic_features(self, df, cache_key_prefix=None):
        """Generates frozen dense embeddings."""
        text_series = self.text_processor.process(df)
        # Use cache if key provided
        cache_key = f"{cache_key_prefix}_embeddings" if cache_key_prefix else None
        embeddings = self.sentence_embedder.transform(text_series, cache_key=cache_key)
        return embeddings

    def _get_augmented_metadata(self, df, injector=None, augmenter=None, fit=False):
        """Generates NMF-augmented metadata."""
        if fit:
            injector = LatentCommunityInjector()
            injector.fit(df)
            nmf_features = injector.transform(df)

            augmenter = MetadataAugmenter()
            augmenter.fit(df)
            meta_features = augmenter.transform(df, nmf_features)

            return meta_features, injector, augmenter
        else:
            if injector is None or augmenter is None:
                raise ValueError("Injector and Augmenter must be provided if fit=False")

            nmf_features = injector.transform(df)
            meta_features = augmenter.transform(df, nmf_features)
            return meta_features

    def train_oof(self, debug=False):
        """
        Performs 5-Fold Stratified CV to generate OOF predictions.
        """
        print("Loading data for OOF Training...")
        train_df, val_df, _ = load_datasets(debug=debug)

        # Combine for CV
        full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        y = get_target(full_df).values

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Initialize OOF matrix: (n_samples, 6 learners)
        oof_preds = np.zeros((len(full_df), 6))

        print(f"Starting {Config.N_FOLDS}-Fold CV...")

        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y)):
            print(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            # Split Data
            X_train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
            X_val_fold_df = full_df.iloc[val_idx].reset_index(drop=True)
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # ---------------------------------------------------------
            # Feature Engineering (Fit on Train Fold ONLY)
            # ---------------------------------------------------------

            # 1. Metadata & NMF
            meta_train, injector_fold, augmenter_fold = self._get_augmented_metadata(
                X_train_fold_df, fit=True
            )
            meta_val = self._get_augmented_metadata(
                X_val_fold_df,
                injector=injector_fold,
                augmenter=augmenter_fold,
                fit=False,
            )

            # Scale Metadata (Dense)
            scaler_fold = StandardScaler()
            meta_train = scaler_fold.fit_transform(meta_train)
            meta_val = scaler_fold.transform(meta_val)

            # 2. Lexical (Text TF-IDF)
            lex_train, lex_vectorizer_fold = self._get_sparse_text_features(
                X_train_fold_df, fit=True
            )
            lex_val = self._get_sparse_text_features(
                X_val_fold_df, vectorizer=lex_vectorizer_fold, fit=False
            )

            # 3. Community (Subreddit TF-IDF)
            comm_train, comm_vectorizer_fold = self._get_sparse_community_features(
                X_train_fold_df, fit=True
            )
            comm_val = self._get_sparse_community_features(
                X_val_fold_df, vectorizer=comm_vectorizer_fold, fit=False
            )

            # 4. Semantic (Embeddings) - No fitting needed, just transform
            # We use cache keys based on indices to avoid recomputing if possible,
            # but for simplicity in CV loop we might just compute.
            # In debug mode, data is small.
            sem_train = self._get_dense_semantic_features(X_train_fold_df)
            sem_val = self._get_dense_semantic_features(X_val_fold_df)

            # ---------------------------------------------------------
            # Construct Feature Sets for each Branch
            # ---------------------------------------------------------

            # Branch 1: Lexical (Sparse Text + Dense Meta)
            X_lex_train = sparse.hstack([lex_train, meta_train]).tocsr()
            X_lex_val = sparse.hstack([lex_val, meta_val]).tocsr()

            # Branch 2: Behavioral (Sparse Community + Dense Meta)
            X_comm_train = sparse.hstack([comm_train, meta_train]).tocsr()
            X_comm_val = sparse.hstack([comm_val, meta_val]).tocsr()

            # Branch 3: Semantic (Dense Embeddings + Dense Meta)
            X_sem_train = np.hstack([sem_train, meta_train])
            X_sem_val = np.hstack([sem_val, meta_val])

            # Branch 4: Contextual (Dense Meta Only)
            X_meta_train = meta_train
            X_meta_val = meta_val

            # ---------------------------------------------------------
            # Train Base Learners
            # ---------------------------------------------------------

            # Calculate scale_pos_weight for XGB/LGBM
            n_pos = np.sum(y_train_fold)
            n_neg = len(y_train_fold) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            learners_map = [
                ("lexical_bagger", X_lex_train, X_lex_val),
                ("community_bagger", X_comm_train, X_comm_val),
                ("semantic_booster", X_sem_train, X_sem_val),
                ("semantic_gradient", X_sem_train, X_sem_val),
                ("semantic_bagger", X_sem_train, X_sem_val),
                ("metadata_anchor", X_meta_train, X_meta_val),
            ]

            for i, (name, X_tr, X_v) in enumerate(learners_map):
                model = get_base_learner(name)

                # Apply model-specific fit params
                if name == "semantic_booster":  # XGB
                    model.set_params(scale_pos_weight=scale_pos_weight)
                    model.fit(
                        X_tr, y_train_fold, eval_set=[(X_v, y_val_fold)], verbose=False
                    )
                elif name == "semantic_gradient":  # LGBM
                    model.set_params(scale_pos_weight=scale_pos_weight)
                    # LGBM uses callbacks for early stopping in recent versions or specific params
                    # We rely on configured params. Note: sklearn API fit accepts eval_set
                    # We suppress output via verbose=-1 in config
                    model.fit(X_tr, y_train_fold, eval_set=[(X_v, y_val_fold)])
                else:
                    model.fit(X_tr, y_train_fold)

                # Predict
                preds = model.predict_proba(X_v)[:, 1]
                oof_preds[val_idx, i] = preds

        # Calculate OOF Score
        overall_auc = roc_auc_score(
            y, oof_preds.mean(axis=1)
        )  # Simple average for quick check
        print(f"OOF Generation Complete. Simple Average AUC: {overall_auc}")

        return oof_preds, y

    def train_meta(self, oof_preds, y):
        """Trains the Level 2 Meta Learner."""
        print("Training Meta Learner...")
        self.meta_learner = get_meta_learner()
        self.meta_learner.fit(oof_preds, y)

        # Check coefficients
        print(f"Meta Learner Coefficients: {self.meta_learner.coef_}")
        return self.meta_learner

    def retrain_final(self, debug=False):
        """
        Validation-Guided Retraining.
        Fits transformers on Full Train.
        Retrains models appropriately.
        """
        print("Starting Final Retraining...")
        train_df, val_df, _ = load_datasets(debug=debug)

        # Full dataset for global transformers and RF/Linear models
        full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        y_full = get_target(full_df).values
        y_train = get_target(train_df).values
        y_val = get_target(val_df).values

        # ---------------------------------------------------------
        # 1. Fit Global Transformers (on Full Train)
        # ---------------------------------------------------------
        print("Fitting Global Transformers...")

        # Metadata & NMF
        meta_full, self.final_nmf_injector, self.final_meta_augmenter = (
            self._get_augmented_metadata(full_df, fit=True)
        )
        self.final_scaler.fit(meta_full)

        # Lexical TF-IDF
        _, self.final_text_tfidf = self._get_sparse_text_features(full_df, fit=True)

        # Community TF-IDF
        _, self.final_community_tfidf = self._get_sparse_community_features(
            full_df, fit=True
        )

        self.final_transformers_fitted = True

        # ---------------------------------------------------------
        # 2. Prepare Data Subsets
        # ---------------------------------------------------------
        # We need:
        # A. Full Transformed Data (for RF/Linear)
        # B. Split Transformed Data (for XGB/LGBM early stopping)

        # Helper to transform a DF using global transformers
        def transform_subset(df):
            # Meta
            m = self._get_augmented_metadata(
                df,
                injector=self.final_nmf_injector,
                augmenter=self.final_meta_augmenter,
                fit=False,
            )
            m = self.final_scaler.transform(m)
            # Lexical
            l = self._get_sparse_text_features(
                df, vectorizer=self.final_text_tfidf, fit=False
            )
            # Community
            c = self._get_sparse_community_features(
                df, vectorizer=self.final_community_tfidf, fit=False
            )
            # Semantic
            s = self._get_dense_semantic_features(df)  # cached internally if run before
            return m, l, c, s

        # Transform Full
        m_full, l_full, c_full, s_full = transform_subset(full_df)

        # Transform Splits (using global transformers)
        m_train, l_train, c_train, s_train = transform_subset(train_df)
        m_val, l_val, c_val, s_val = transform_subset(val_df)

        # Construct Branch Features (Full)
        X_lex_full = sparse.hstack([l_full, m_full]).tocsr()
        X_comm_full = sparse.hstack([c_full, m_full]).tocsr()
        X_sem_full = np.hstack([s_full, m_full])
        X_meta_full = m_full

        # Construct Branch Features (Split - for GBMs)
        X_sem_train = np.hstack([s_train, m_train])
        X_sem_val = np.hstack([s_val, m_val])

        # ---------------------------------------------------------
        # 3. Retrain Base Learners
        # ---------------------------------------------------------

        # Calculate scale_pos_weight
        n_pos = np.sum(y_train)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Define Models
        learners_config = [
            ("lexical_bagger", X_lex_full, y_full, None, None),
            ("community_bagger", X_comm_full, y_full, None, None),
            ("semantic_bagger", X_sem_full, y_full, None, None),
            ("metadata_anchor", X_meta_full, y_full, None, None),
            # GBMs use split data
            ("semantic_booster", X_sem_train, y_train, X_sem_val, y_val),
            ("semantic_gradient", X_sem_train, y_train, X_sem_val, y_val),
        ]

        for name, X, y, X_eval, y_eval in learners_config:
            print(f"Retraining {name}...")
            model = get_base_learner(name)

            if name in ["semantic_booster", "semantic_gradient"]:
                model.set_params(scale_pos_weight=scale_pos_weight)
                model.fit(X, y, eval_set=[(X_eval, y_eval)])
            else:
                model.fit(X, y)

            self.base_learners[name] = model

        print("Final Retraining Complete.")

    def predict(self, debug=False):
        """
        Generates predictions for the test set.
        """
        if not self.final_transformers_fitted:
            raise RuntimeError("Must run retrain_final before predict.")

        print("Loading Test Data...")
        _, _, test_df = load_datasets(debug=debug)

        # Transform Test Data
        print("Transforming Test Data...")
        # Meta
        m_test = self._get_augmented_metadata(
            test_df,
            injector=self.final_nmf_injector,
            augmenter=self.final_meta_augmenter,
            fit=False,
        )
        m_test = self.final_scaler.transform(m_test)
        # Lexical
        l_test = self._get_sparse_text_features(
            test_df, vectorizer=self.final_text_tfidf, fit=False
        )
        # Community
        c_test = self._get_sparse_community_features(
            test_df, vectorizer=self.final_community_tfidf, fit=False
        )
        # Semantic
        s_test = self._get_dense_semantic_features(test_df, cache_key_prefix="test")

        # Construct Branch Features
        X_lex = sparse.hstack([l_test, m_test]).tocsr()
        X_comm = sparse.hstack([c_test, m_test]).tocsr()
        X_sem = np.hstack([s_test, m_test])
        X_meta = m_test

        # Generate Level 1 Predictions
        print("Generating Base Predictions...")
        n_samples = len(test_df)
        l1_preds = np.zeros((n_samples, 6))

        # Order must match train_oof and learners_config list order implicitly
        # Map names to indices based on the order used in train_oof
        # 0: lexical, 1: community, 2: booster, 3: gradient, 4: sem_bagger, 5: meta_anchor

        l1_preds[:, 0] = self.base_learners["lexical_bagger"].predict_proba(X_lex)[:, 1]
        l1_preds[:, 1] = self.base_learners["community_bagger"].predict_proba(X_comm)[
            :, 1
        ]
        l1_preds[:, 2] = self.base_learners["semantic_booster"].predict_proba(X_sem)[
            :, 1
        ]
        l1_preds[:, 3] = self.base_learners["semantic_gradient"].predict_proba(X_sem)[
            :, 1
        ]
        l1_preds[:, 4] = self.base_learners["semantic_bagger"].predict_proba(X_sem)[
            :, 1
        ]
        l1_preds[:, 5] = self.base_learners["metadata_anchor"].predict_proba(X_meta)[
            :, 1
        ]

        # Level 2 Prediction
        print("Generating Final Predictions...")
        final_probs = self.meta_learner.predict_proba(l1_preds)[:, 1]

        # Save Submission
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": final_probs,
            }
        )

        Config.ensure_directories()
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
