import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, save_model, print_metrics
from library.data_manager import DataManager
from library.feature_engine import (
    GranularLexicalVectorizer,
    CommunityVectorizer,
    MetadataScaler,
    SemanticEmbedder,
)
from library.model_zoo import (
    get_lexical_bagger,
    get_community_bagger,
    get_semantic_booster,
    get_semantic_gradient,
    get_semantic_bagger,
    get_metadata_anchor,
    get_temporal_booster,
)


class TrainingEngine:
    """
    Orchestrates the training of Level 1 Base Learners using a 5-Fold Stratified CV.
    Implements the 'Volatile vs Stable' training protocol and generates OOF predictions.
    """

    def __init__(self):
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize Feature Engines
        self.lexical_vectorizer = GranularLexicalVectorizer()
        self.community_vectorizer = CommunityVectorizer()
        self.metadata_scaler = MetadataScaler()
        self.semantic_embedder = SemanticEmbedder()

    def run(self):
        """
        Main execution method:
        1. Loads and processes data.
        2. Fits feature engines on the Union dataset.
        3. Runs 5-Fold CV to train models and generate OOFs.
        4. Retrains Stable models on the full dataset.
        """
        set_seed(Config.SEED)

        print("Loading Union Data (Train + Val)...")
        dm = DataManager()
        train_df, _ = dm.load_union_data()

        target_col = "requester_received_pizza"
        if target_col not in train_df.columns:
            raise ValueError(
                f"Target column '{target_col}' missing from training data."
            )

        y = train_df[target_col].values.astype(int)

        # --- Feature Engineering (Fit on Union Data) ---
        print("Fitting Feature Engines on Full Union Data...")

        # 1. Metadata
        self.metadata_scaler.fit(train_df)
        X_meta = self.metadata_scaler.transform(train_df)
        save_model(
            self.metadata_scaler,
            os.path.join(self.models_dir, "metadata_scaler.joblib"),
        )

        # 2. Lexical (Sparse)
        self.lexical_vectorizer.fit(train_df)
        X_lex = self.lexical_vectorizer.transform(train_df)
        save_model(
            self.lexical_vectorizer,
            os.path.join(self.models_dir, "lexical_vectorizer.joblib"),
        )

        # 3. Community (Sparse)
        self.community_vectorizer.fit(train_df)
        X_comm = self.community_vectorizer.transform(train_df)
        save_model(
            self.community_vectorizer,
            os.path.join(self.models_dir, "community_vectorizer.joblib"),
        )

        # 4. Semantic (Dense)
        # Embedder handles its own caching
        X_sem = self.semantic_embedder.transform(train_df, "train_union")

        # --- Construct Feature Matrices ---
        print("Constructing Modality-Specific Feature Matrices...")

        # Sparse Concatenation: TF-IDF + Scaled Metadata
        X_lexical_full = sp.hstack([X_lex, X_meta], format="csr")
        X_community_full = sp.hstack([X_comm, X_meta], format="csr")

        # Dense Concatenation: Embeddings + Scaled Metadata
        X_semantic_full = np.hstack([X_sem, X_meta])

        # Contextual: Metadata only
        X_meta_full = X_meta

        # --- 5-Fold Stratified CV ---
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Initialize OOF Prediction Containers
        oof_preds = {
            "lexical_bagger": np.zeros(len(y)),
            "community_bagger": np.zeros(len(y)),
            "semantic_booster": np.zeros(len(y)),
            "semantic_gradient": np.zeros(len(y)),
            "semantic_bagger": np.zeros(len(y)),
            "metadata_anchor": np.zeros(len(y)),
            "temporal_booster": np.zeros(len(y)),
        }

        print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
            print(f"--- Processing Fold {fold} ---")

            # Target Split
            y_tr, y_val = y[train_idx], y[val_idx]

            # ------------------------------------------------------------------
            # 1. Sparse Lexical Branch
            # ------------------------------------------------------------------
            X_tr = X_lexical_full[train_idx]
            X_val = X_lexical_full[val_idx]

            # Stable: Lexical Bagger (RF)
            model = get_lexical_bagger()
            model.fit(X_tr, y_tr)
            oof_preds["lexical_bagger"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"lexical_bagger_fold_{fold}.joblib"),
            )

            # ------------------------------------------------------------------
            # 2. Sparse Behavioral Branch
            # ------------------------------------------------------------------
            X_tr = X_community_full[train_idx]
            X_val = X_community_full[val_idx]

            # Stable: Community Bagger (RF)
            model = get_community_bagger()
            model.fit(X_tr, y_tr)
            oof_preds["community_bagger"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"community_bagger_fold_{fold}.joblib"),
            )

            # ------------------------------------------------------------------
            # 3. Dense Semantic Branch
            # ------------------------------------------------------------------
            X_tr = X_semantic_full[train_idx]
            X_val = X_semantic_full[val_idx]

            # Volatile: Semantic Booster (XGB)
            # Uses Early Stopping
            model = get_semantic_booster()
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_preds["semantic_booster"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"semantic_booster_fold_{fold}.joblib"),
            )

            # Volatile: Semantic Gradient (LGBM)
            # Uses Early Stopping
            model = get_semantic_gradient()
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            oof_preds["semantic_gradient"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"semantic_gradient_fold_{fold}.joblib"),
            )

            # Stable: Semantic Bagger (RF)
            model = get_semantic_bagger()
            model.fit(X_tr, y_tr)
            oof_preds["semantic_bagger"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"semantic_bagger_fold_{fold}.joblib"),
            )

            # ------------------------------------------------------------------
            # 4. Contextual Branch
            # ------------------------------------------------------------------
            X_tr = X_meta_full[train_idx]
            X_val = X_meta_full[val_idx]

            # Stable: Metadata Anchor (LR)
            model = get_metadata_anchor()
            model.fit(X_tr, y_tr)
            oof_preds["metadata_anchor"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"metadata_anchor_fold_{fold}.joblib"),
            )

            # Volatile: Temporal Booster (LGBM)
            model = get_temporal_booster()
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            oof_preds["temporal_booster"][val_idx] = model.predict_proba(X_val)[:, 1]
            save_model(
                model,
                os.path.join(self.models_dir, f"temporal_booster_fold_{fold}.joblib"),
            )

        # --- Save OOF Predictions ---
        oof_df = pd.DataFrame(oof_preds)
        oof_df[target_col] = y
        oof_df.to_csv(self.oof_path, index=False)
        print(f"OOF predictions saved to {self.oof_path}")

        # --- Print Metrics ---
        print("\n=== OOF Performance (AUC) ===")
        metrics = {}
        for col in oof_preds:
            auc = roc_auc_score(y, oof_preds[col])
            metrics[f"{col}_AUC"] = auc
        print_metrics(metrics)

        # --- Retrain Stable Models on Full Union Data ---
        print("\nRetraining Stable Models on Full Union Data...")

        # 1. Lexical Bagger
        model = get_lexical_bagger()
        model.fit(X_lexical_full, y)
        save_model(model, os.path.join(self.models_dir, "lexical_bagger_full.joblib"))

        # 2. Community Bagger
        model = get_community_bagger()
        model.fit(X_community_full, y)
        save_model(model, os.path.join(self.models_dir, "community_bagger_full.joblib"))

        # 3. Semantic Bagger
        model = get_semantic_bagger()
        model.fit(X_semantic_full, y)
        save_model(model, os.path.join(self.models_dir, "semantic_bagger_full.joblib"))

        # 4. Metadata Anchor
        model = get_metadata_anchor()
        model.fit(X_meta_full, y)
        save_model(model, os.path.join(self.models_dir, "metadata_anchor_full.joblib"))

        print("Training Pipeline Complete.")
