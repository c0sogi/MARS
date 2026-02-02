import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from library.config import Config
from library.utils import set_seed, load_model
from library.data_manager import DataManager
from library.feature_engine import (
    GranularLexicalVectorizer,
    CommunityVectorizer,
    MetadataScaler,
    SemanticEmbedder,
)
from library.model_zoo import get_meta_learner


class InferenceEngine:
    """
    Orchestrates the inference process using the Consistent Hybrid Inference Protocol.
    Generates Level 1 predictions (Hybrid: Full-Fit for Stable, CV-Bagging for Volatile),
    trains the Level 2 Meta-Learner on OOFs, and produces the final submission.
    """

    def __init__(self):
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        self.oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
        self.submission_path = Config.SUBMISSION_PATH

        # Initialize Feature Engines
        # Note: We will load fitted states from disk, but need instances to call transform
        self.lexical_vectorizer = GranularLexicalVectorizer()
        self.community_vectorizer = CommunityVectorizer()
        self.metadata_scaler = MetadataScaler()
        self.semantic_embedder = SemanticEmbedder()

    def run(self):
        """
        Main execution method for inference.
        """
        set_seed(Config.SEED)
        print("Starting Inference Pipeline...")

        # ----------------------------------------------------------------------
        # 1. Load and Transform Test Data
        # ----------------------------------------------------------------------
        print("Loading and Transforming Test Data...")
        dm = DataManager()
        _, test_df = dm.load_union_data()

        # Load fitted feature engines
        self.metadata_scaler = load_model(
            os.path.join(self.models_dir, "metadata_scaler.joblib")
        )
        self.lexical_vectorizer = load_model(
            os.path.join(self.models_dir, "lexical_vectorizer.joblib")
        )
        self.community_vectorizer = load_model(
            os.path.join(self.models_dir, "community_vectorizer.joblib")
        )

        # Transform Data
        # 1. Metadata
        X_meta = self.metadata_scaler.transform(test_df)

        # 2. Lexical (Sparse)
        X_lex = self.lexical_vectorizer.transform(test_df)

        # 3. Community (Sparse)
        X_comm = self.community_vectorizer.transform(test_df)

        # 4. Semantic (Dense)
        # Embedder handles caching internally
        X_sem = self.semantic_embedder.transform(test_df, "test")

        # Construct Feature Matrices (Must match TrainingEngine exactly)
        X_lexical_full = sp.hstack([X_lex, X_meta], format="csr")
        X_community_full = sp.hstack([X_comm, X_meta], format="csr")
        X_semantic_full = np.hstack([X_sem, X_meta])
        X_meta_full = X_meta

        # ----------------------------------------------------------------------
        # 2. Level 1 Inference (Hybrid Protocol)
        # ----------------------------------------------------------------------
        print("Generating Level 1 Predictions...")

        # Container for Level 1 Test Predictions
        l1_preds = {}
        n_samples = len(test_df)

        # --- Stable Branches (Load Single Full-Fit Model) ---

        # 1. Lexical Bagger (RF)
        print("Predicting: Lexical Bagger (Stable)...")
        model = load_model(os.path.join(self.models_dir, "lexical_bagger_full.joblib"))
        l1_preds["lexical_bagger"] = model.predict_proba(X_lexical_full)[:, 1]

        # 2. Community Bagger (RF)
        print("Predicting: Community Bagger (Stable)...")
        model = load_model(
            os.path.join(self.models_dir, "community_bagger_full.joblib")
        )
        l1_preds["community_bagger"] = model.predict_proba(X_community_full)[:, 1]

        # 3. Semantic Bagger (RF)
        print("Predicting: Semantic Bagger (Stable)...")
        model = load_model(os.path.join(self.models_dir, "semantic_bagger_full.joblib"))
        l1_preds["semantic_bagger"] = model.predict_proba(X_semantic_full)[:, 1]

        # 4. Metadata Anchor (LR)
        print("Predicting: Metadata Anchor (Stable)...")
        model = load_model(os.path.join(self.models_dir, "metadata_anchor_full.joblib"))
        l1_preds["metadata_anchor"] = model.predict_proba(X_meta_full)[:, 1]

        # --- Volatile Branches (CV-Bagging: Average of 5 Folds) ---

        # Initialize accumulators
        sem_booster_accum = np.zeros(n_samples)
        sem_gradient_accum = np.zeros(n_samples)
        temp_booster_accum = np.zeros(n_samples)

        print(
            f"Predicting: Volatile Models (CV-Bagging over {Config.N_FOLDS} folds)..."
        )

        for fold in range(Config.N_FOLDS):
            # Semantic Booster (XGB)
            model = load_model(
                os.path.join(self.models_dir, f"semantic_booster_fold_{fold}.joblib")
            )
            sem_booster_accum += model.predict_proba(X_semantic_full)[:, 1]

            # Semantic Gradient (LGBM)
            model = load_model(
                os.path.join(self.models_dir, f"semantic_gradient_fold_{fold}.joblib")
            )
            sem_gradient_accum += model.predict_proba(X_semantic_full)[:, 1]

            # Temporal Booster (LGBM)
            model = load_model(
                os.path.join(self.models_dir, f"temporal_booster_fold_{fold}.joblib")
            )
            temp_booster_accum += model.predict_proba(X_meta_full)[:, 1]

        # Average predictions
        l1_preds["semantic_booster"] = sem_booster_accum / Config.N_FOLDS
        l1_preds["semantic_gradient"] = sem_gradient_accum / Config.N_FOLDS
        l1_preds["temporal_booster"] = temp_booster_accum / Config.N_FOLDS

        # ----------------------------------------------------------------------
        # 3. Level 2 Meta-Learning
        # ----------------------------------------------------------------------
        print("Training Meta-Learner on OOFs...")

        # Load OOFs
        if not os.path.exists(self.oof_path):
            raise FileNotFoundError(f"OOF predictions not found at {self.oof_path}")

        oof_df = pd.read_csv(self.oof_path)

        # Define feature order to ensure consistency
        feature_cols = [
            "lexical_bagger",
            "community_bagger",
            "semantic_booster",
            "semantic_gradient",
            "semantic_bagger",
            "metadata_anchor",
            "temporal_booster",
        ]

        # Prepare Train (OOF) and Test (L1 Preds) Matrices
        X_meta_train = oof_df[feature_cols].values
        y_meta_train = oof_df["requester_received_pizza"].values

        # Construct test dataframe to ensure column order matches
        test_l1_df = pd.DataFrame(l1_preds)
        X_meta_test = test_l1_df[feature_cols].values

        # Train Meta-Learner
        meta_learner = get_meta_learner()
        meta_learner.fit(X_meta_train, y_meta_train)

        # Predict Final Probabilities
        print("Generating Final Predictions...")
        final_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

        # ----------------------------------------------------------------------
        # 4. Save Submission
        # ----------------------------------------------------------------------
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": final_probs,
            }
        )

        submission_df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
        print(f"Submission Shape: {submission_df.shape}")
        print("Inference Pipeline Complete.")
