import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import Timer, print_metric, set_seed
from library.data_processing import process_data
from library.feature_extraction import FeatureFactory
from library.model_definitions import (
    LexicalBagger,
    CommunityBagger,
    SemanticBooster,
    SemanticBagger,
    InteractionBooster,
    MetadataAnchor,
    MetaLearner,
)


class CrossValidator:
    """
    Handles Stratified K-Fold Cross Validation to generate Out-Of-Fold (OOF) predictions
    for the Level 1 models.
    """

    def __init__(self, n_folds=Config.N_FOLDS, random_state=Config.RANDOM_SEED):
        self.n_folds = n_folds
        self.skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    def _slice_features(self, features_dict, indices):
        """Helper to slice dictionary of features (sparse or dense)."""
        sliced = {}
        for key, data in features_dict.items():
            sliced[key] = data[indices]
        return sliced

    def run_cv(self, X_train_dict, y_train):
        """
        Performs CV on the training set.

        Args:
            X_train_dict (dict): Dictionary of feature matrices for the training set.
            y_train (np.array): Target labels for the training set.

        Returns:
            np.array: OOF predictions matrix (n_samples, n_models).
        """
        # Initialize OOF matrix: (n_samples, 6 models)
        # Order: Lexical, Community, SemBoost, SemBag, Interaction, Anchor
        oof_preds = np.zeros((len(y_train), 6))

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(
            self.skf.split(np.zeros(len(y_train)), y_train)
        ):
            with Timer(f"Fold {fold + 1}"):
                # Slice data
                X_fold_train = self._slice_features(X_train_dict, train_idx)
                X_fold_val = self._slice_features(X_train_dict, val_idx)
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]

                # --- 1. Lexical Bagger ---
                model_lex = LexicalBagger()
                model_lex.fit(
                    X_fold_train["lexical"], X_fold_train["metadata"], y_fold_train
                )
                p_lex = model_lex.predict_proba(
                    X_fold_val["lexical"], X_fold_val["metadata"]
                )[:, 1]
                oof_preds[val_idx, 0] = p_lex

                # --- 2. Community Bagger ---
                model_com = CommunityBagger()
                model_com.fit(
                    X_fold_train["behavioral"], X_fold_train["metadata"], y_fold_train
                )
                p_com = model_com.predict_proba(
                    X_fold_val["behavioral"], X_fold_val["metadata"]
                )[:, 1]
                oof_preds[val_idx, 1] = p_com

                # --- 3. Semantic Booster (XGB) ---
                # Uses fold validation set for early stopping
                model_sem_boost = SemanticBooster()
                model_sem_boost.fit(
                    X_fold_train["semantic"],
                    X_fold_train["metadata"],
                    y_fold_train,
                    X_semantic_val=X_fold_val["semantic"],
                    X_metadata_val=X_fold_val["metadata"],
                    y_val=y_fold_val,
                )
                p_sem_boost = model_sem_boost.predict_proba(
                    X_fold_val["semantic"], X_fold_val["metadata"]
                )[:, 1]
                oof_preds[val_idx, 2] = p_sem_boost

                # --- 4. Semantic Bagger (RF) ---
                model_sem_bag = SemanticBagger()
                model_sem_bag.fit(
                    X_fold_train["semantic"], X_fold_train["metadata"], y_fold_train
                )
                p_sem_bag = model_sem_bag.predict_proba(
                    X_fold_val["semantic"], X_fold_val["metadata"]
                )[:, 1]
                oof_preds[val_idx, 3] = p_sem_bag

                # --- 5. Interaction Booster (LGBM) ---
                # Uses fold validation set for early stopping
                model_inter = InteractionBooster()
                model_inter.fit(
                    X_fold_train["interaction"],
                    y_fold_train,
                    X_interaction_val=X_fold_val["interaction"],
                    y_val=y_fold_val,
                )
                p_inter = model_inter.predict_proba(X_fold_val["interaction"])[:, 1]
                oof_preds[val_idx, 4] = p_inter

                # --- 6. Metadata Anchor ---
                model_meta = MetadataAnchor()
                model_meta.fit(X_fold_train["metadata"], y_fold_train)
                p_meta = model_meta.predict_proba(X_fold_val["metadata"])[:, 1]
                oof_preds[val_idx, 5] = p_meta

        return oof_preds


class FinalRetrainer:
    """
    Handles the final retraining of models using the Validation-Guided Protocol
    and generates test predictions.
    """

    def __init__(self):
        self.meta_learner = MetaLearner()
        self.models = {
            "lexical": LexicalBagger(),
            "community": CommunityBagger(),
            "sem_boost": SemanticBooster(),
            "sem_bag": SemanticBagger(),
            "interaction": InteractionBooster(),
            "anchor": MetadataAnchor(),
        }

    def _concat_features(self, train_data, val_data, is_sparse=False):
        if is_sparse:
            return sp.vstack([train_data, val_data], format="csr")
        else:
            return np.vstack([train_data, val_data])

    def run(self, X_train_dict, y_train, X_val_dict, y_val, X_test_dict, oof_preds):
        """
        Retrains models and predicts on test set.
        """
        print("\n=== Training Meta-Learner ===")
        # Train Meta-Learner on OOF predictions
        self.meta_learner.fit(oof_preds, y_train)

        # Calculate OOF Score
        oof_score = roc_auc_score(
            y_train, self.meta_learner.predict_proba(oof_preds)[:, 1]
        )
        print_metric("Meta-Learner OOF AUC", oof_score)

        print("\n=== Retraining Level 1 Models ===")

        # Prepare Full Training Data (Train + Val) for Baggers/Linear
        X_full_lex = self._concat_features(
            X_train_dict["lexical"], X_val_dict["lexical"], is_sparse=True
        )
        X_full_beh = self._concat_features(
            X_train_dict["behavioral"], X_val_dict["behavioral"], is_sparse=True
        )
        X_full_sem = self._concat_features(
            X_train_dict["semantic"], X_val_dict["semantic"], is_sparse=False
        )
        X_full_meta = self._concat_features(
            X_train_dict["metadata"], X_val_dict["metadata"], is_sparse=False
        )
        y_full = np.concatenate([y_train, y_val])

        # 1. Lexical Bagger (Full Data)
        self.models["lexical"].fit(X_full_lex, X_full_meta, y_full)

        # 2. Community Bagger (Full Data)
        self.models["community"].fit(X_full_beh, X_full_meta, y_full)

        # 3. Semantic Bagger (Full Data)
        self.models["sem_bag"].fit(X_full_sem, X_full_meta, y_full)

        # 4. Metadata Anchor (Full Data)
        self.models["anchor"].fit(X_full_meta, y_full)

        # 5. Semantic Booster (Train w/ Val Early Stopping)
        self.models["sem_boost"].fit(
            X_train_dict["semantic"],
            X_train_dict["metadata"],
            y_train,
            X_semantic_val=X_val_dict["semantic"],
            X_metadata_val=X_val_dict["metadata"],
            y_val=y_val,
        )

        # 6. Interaction Booster (Train w/ Val Early Stopping)
        self.models["interaction"].fit(
            X_train_dict["interaction"],
            y_train,
            X_interaction_val=X_val_dict["interaction"],
            y_val=y_val,
        )

        print("\n=== Generating Test Predictions ===")
        # Initialize Test Prediction Matrix
        test_preds_l1 = np.zeros((X_test_dict["metadata"].shape[0], 6))

        test_preds_l1[:, 0] = self.models["lexical"].predict_proba(
            X_test_dict["lexical"], X_test_dict["metadata"]
        )[:, 1]
        test_preds_l1[:, 1] = self.models["community"].predict_proba(
            X_test_dict["behavioral"], X_test_dict["metadata"]
        )[:, 1]
        test_preds_l1[:, 2] = self.models["sem_boost"].predict_proba(
            X_test_dict["semantic"], X_test_dict["metadata"]
        )[:, 1]
        test_preds_l1[:, 3] = self.models["sem_bag"].predict_proba(
            X_test_dict["semantic"], X_test_dict["metadata"]
        )[:, 1]
        test_preds_l1[:, 4] = self.models["interaction"].predict_proba(
            X_test_dict["interaction"]
        )[:, 1]
        test_preds_l1[:, 5] = self.models["anchor"].predict_proba(
            X_test_dict["metadata"]
        )[:, 1]

        # Final Stacking
        final_probs = self.meta_learner.predict_proba(test_preds_l1)[:, 1]

        return final_probs


def run_workflow(load_cached_data=True):
    """
    Main execution pipeline.
    """
    set_seed(Config.RANDOM_SEED)
    Config.ensure_directories()

    # 1. Load Data
    train_df, val_df, test_df = process_data(load_cached_data=load_cached_data)

    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # 2. Extract Features
    ff = FeatureFactory()

    # Lexical
    X_train_lex, X_val_lex, X_test_lex = ff.get_lexical_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Behavioral
    X_train_beh, X_val_beh, X_test_beh = ff.get_behavioral_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Semantic
    X_train_sem, X_val_sem, X_test_sem = ff.get_semantic_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Metadata
    X_train_meta, X_val_meta, X_test_meta = ff.get_metadata_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
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
        load_cached_data=load_cached_data,
    )

    # Pack into dictionaries
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

    # 3. Cross Validation
    cv = CrossValidator()
    oof_preds = cv.run_cv(X_train_dict, y_train)

    # 4. Final Retraining and Inference
    retrainer = FinalRetrainer()
    final_probs = retrainer.run(
        X_train_dict, y_train, X_val_dict, y_val, X_test_dict, oof_preds
    )

    # 5. Save Submission
    submission = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": final_probs}
    )

    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Workflow completed successfully.")
