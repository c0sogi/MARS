import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config
from library.utils import seed_everything


class BaseDiscriminantExpert:
    """
    Wrapper for Linear Discriminant Analysis with specific solver and shrinkage settings.
    Acts as a base expert for a specific modality.
    """

    def __init__(self, solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE):
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, X, y):
        """
        Fits the LDA model.
        """
        self.model.fit(X, y)
        return self

    def predict_logits(self, X):
        """
        Returns the decision function values (logits/log-posterior ratios).
        Shape: (N, n_classes)
        """
        return self.model.decision_function(X)

    def predict_proba(self, X):
        """
        Returns probability estimates.
        """
        return self.model.predict_proba(X)


class StackedEnsemble:
    """
    Two-Level Stacked Ensemble with Hyper-Densified OOF Projection.

    Level 1: Base Discriminant Experts (DINO, ConvNeXt, Tabular)
    Level 2: Meta Discriminant Expert (LDA on concatenated logits)
    """

    def __init__(self):
        seed_everything(Config.SEED)

        # Level 1 Experts
        self.expert_dino = BaseDiscriminantExpert()
        self.expert_conv = BaseDiscriminantExpert()
        self.expert_tab = BaseDiscriminantExpert()

        # Level 2 Meta-Learner
        # Uses specific meta-solver settings from Config
        self.meta_learner = BaseDiscriminantExpert(
            solver=Config.META_SOLVER, shrinkage=Config.META_SHRINKAGE
        )

        self.n_classes = None

    def fit(self, X_dino, X_conv, X_tab, y, ids):
        """
        Trains the stacked ensemble using OOF stacking.

        Args:
            X_dino, X_conv, X_tab: Hyper-densified training features (N*9 samples).
            y: Labels corresponding to densified features.
            ids: Image IDs corresponding to densified features (used for grouping).
        """
        print("Training Stacked Ensemble...")

        # 1. Setup for Inner CV (Grouped by ID to prevent leakage)
        # We need to split based on unique images, not individual densified views.
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        unique_labels = y[unique_indices]

        self.n_classes = len(np.unique(unique_labels))

        # Prepare arrays to store OOF logits
        # Shape: (Total_Samples, n_classes)
        n_samples = X_dino.shape[0]
        oof_logits_dino = np.zeros((n_samples, self.n_classes))
        oof_logits_conv = np.zeros((n_samples, self.n_classes))
        oof_logits_tab = np.zeros((n_samples, self.n_classes))

        # Stratified K-Fold on Unique IDs
        skf = StratifiedKFold(
            n_splits=Config.INNER_FOLDS, shuffle=True, random_state=Config.SEED
        )

        print(f"Generating OOF Logits via {Config.INNER_FOLDS}-Fold CV...")

        for fold, (train_idx_unique, val_idx_unique) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            # Map unique ID indices back to full densified data indices
            train_ids_set = set(unique_ids[train_idx_unique])
            val_ids_set = set(unique_ids[val_idx_unique])

            # Boolean masks for the full densified arrays
            # This is faster than list comprehensions for large arrays
            train_mask = np.isin(ids, list(train_ids_set))
            val_mask = np.isin(ids, list(val_ids_set))

            # Split Data
            X_d_tr, X_d_val = X_dino[train_mask], X_dino[val_mask]
            X_c_tr, X_c_val = X_conv[train_mask], X_conv[val_mask]
            X_t_tr, X_t_val = X_tab[train_mask], X_tab[val_mask]
            y_tr = y[train_mask]

            # Train Base Experts on Inner Fold
            dino_model = BaseDiscriminantExpert().fit(X_d_tr, y_tr)
            conv_model = BaseDiscriminantExpert().fit(X_c_tr, y_tr)
            tab_model = BaseDiscriminantExpert().fit(X_t_tr, y_tr)

            # Predict Logits on Validation Part
            # Store in the corresponding slots of the OOF arrays
            # Note: val_mask is boolean, so we can index directly
            oof_logits_dino[val_mask] = dino_model.predict_logits(X_d_val)
            oof_logits_conv[val_mask] = conv_model.predict_logits(X_c_val)
            oof_logits_tab[val_mask] = tab_model.predict_logits(X_t_val)

        # 2. Train Meta-Learner
        print("Training Meta-Learner on OOF Logits...")
        # Concatenate logits: (N, n_classes * 3)
        X_meta = np.hstack([oof_logits_dino, oof_logits_conv, oof_logits_tab])
        self.meta_learner.fit(X_meta, y)

        # Calculate OOF Metrics
        oof_probs = self.meta_learner.predict_proba(X_meta)
        oof_acc = accuracy_score(y, np.argmax(oof_probs, axis=1))
        oof_loss = log_loss(y, oof_probs)

        print(f"  Meta-Learner OOF Accuracy: {oof_acc:.6f}")
        print(f"  Meta-Learner OOF Log Loss: {oof_loss:.6f}")

        # 3. Retrain Base Experts on Full Data
        print("Retraining Base Experts on Full Training Data...")
        self.expert_dino.fit(X_dino, y)
        self.expert_conv.fit(X_conv, y)
        self.expert_tab.fit(X_tab, y)

        return self

    def predict_proba(self, X_dino, X_conv, X_tab):
        """
        Predicts class probabilities for new data.

        Args:
            X_dino, X_conv, X_tab: Features (N, D).

        Returns:
            np.ndarray: Probabilities (N, n_classes).
        """
        # 1. Get Logits from Base Experts
        logits_dino = self.expert_dino.predict_logits(X_dino)
        logits_conv = self.expert_conv.predict_logits(X_conv)
        logits_tab = self.expert_tab.predict_logits(X_tab)

        # 2. Concatenate Logits
        X_meta = np.hstack([logits_dino, logits_conv, logits_tab])

        # 3. Predict with Meta-Learner
        return self.meta_learner.predict_proba(X_meta)
