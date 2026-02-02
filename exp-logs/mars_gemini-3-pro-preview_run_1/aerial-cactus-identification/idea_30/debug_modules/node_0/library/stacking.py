import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import get_logger, seed_everything
from library.data import cache_dataset_in_ram

logger = get_logger("stacking")


def calculate_trust_score(pred_fsizes, true_fsizes):
    """
    Calculates the Trust Score based on the error in file size prediction.
    Formula: T = exp(-|pred - true|)

    Args:
        pred_fsizes (np.ndarray): Predicted normalized log file sizes.
        true_fsizes (np.ndarray): Ground truth normalized log file sizes.

    Returns:
        np.ndarray: Trust scores in range (0, 1].
    """
    # Ensure inputs are numpy arrays and flattened
    p = np.array(pred_fsizes).flatten()
    t = np.array(true_fsizes).flatten()

    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs true {t.shape}")

    # Calculate absolute error
    abs_error = np.abs(p - t)

    # Calculate exponential trust score
    trust_scores = np.exp(-abs_error)

    return trust_scores


def create_interaction_features(probs, pred_fsizes, true_fsizes):
    """
    Creates interaction features for the meta-learner.
    Feature = Probability * TrustScore

    Args:
        probs (np.ndarray): Class probabilities (N,).
        pred_fsizes (np.ndarray): Predicted file sizes (N,).
        true_fsizes (np.ndarray): True file sizes (N,).

    Returns:
        np.ndarray: Interaction features (N, 1).
    """
    probs = np.array(probs).flatten()
    trust = calculate_trust_score(pred_fsizes, true_fsizes)

    # Interaction: Modulate probability by trust
    # If trust is low, the feature value is pushed towards 0.
    interaction = probs * trust

    return interaction.reshape(-1, 1)


class MetaLearner:
    """
    Logistic Regression Meta-Learner trained on interaction features.
    """

    def __init__(self, random_state=Config.SEED):
        self.model = LogisticRegression(random_state=random_state, solver="liblinear")
        self.is_fitted = False

    def fit(self, X, y):
        """
        Args:
            X (np.ndarray): Feature matrix (N_samples, N_models).
            y (np.ndarray): Target labels (N_samples,).
        """
        logger.info(f"Training MetaLearner on data shape: {X.shape}")
        self.model.fit(X, y)
        self.is_fitted = True

        # Log coefficients for interpretability
        logger.info(f"MetaLearner Coefficients: {self.model.coef_}")
        logger.info(f"MetaLearner Intercept: {self.model.intercept_}")

    def predict(self, X):
        """
        Args:
            X (np.ndarray): Feature matrix (N_samples, N_models).
        Returns:
            np.ndarray: Predicted probabilities (N_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("MetaLearner must be fitted before prediction.")

        # Predict probabilities for class 1
        return self.model.predict_proba(X)[:, 1]


class StackingDataManager:
    """
    Handles data loading and caching for the stacking process.
    """

    def __init__(self, working_dir=Config.WORKING_DIR):
        self.working_dir = working_dir
        self.cache_dir = os.path.join(working_dir, "stacking_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_ground_truth(self):
        """
        Loads ground truth labels and file sizes using library.data.
        """
        logger.info("Loading ground truth data from library.data...")
        # We use load_cached_data=True to leverage existing cache from training
        (tr_data, val_data, te_data) = cache_dataset_in_ram(load_cached_data=True)

        # Unpack
        tr_imgs, tr_lbls, tr_fs, tr_ids = tr_data
        val_imgs, val_lbls, val_fs, val_ids = val_data
        te_imgs, te_lbls, te_fs, te_ids = te_data

        # Combine Train and Val for OOF purposes if models were trained on full CV
        # However, typically OOF corresponds to the entire training set (Train + Val split)
        # We need to ensure the order matches the OOF predictions provided.
        # For this implementation, we assume OOF predictions are aligned with the concatenation of Train + Val
        # or simply the 'train_metadata.csv' + 'val_metadata.csv' order.
        # To be safe and consistent with standard CV, we return the full training set arrays.

        # Concatenate train and val to represent the full development set
        full_train_lbls = np.concatenate([tr_lbls, val_lbls])
        full_train_fs = np.concatenate([tr_fs, val_fs])
        full_train_ids = np.concatenate([tr_ids, val_ids])

        return {
            "train_labels": full_train_lbls,
            "train_fsizes": full_train_fs,
            "train_ids": full_train_ids,
            "test_fsizes": te_fs,
            "test_ids": te_ids,
        }

    def get_features(self, model_preds, true_fsizes, prefix="train", load_cache=True):
        """
        Generates or loads interaction features for a set of models.

        Args:
            model_preds (dict): Dict of {model_name: {'probs': np.array, 'fsizes': np.array}}
            true_fsizes (np.ndarray): Ground truth file sizes.
            prefix (str): 'train' (for OOF) or 'test'.
            load_cache (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Feature matrix (N_samples, N_models).
        """
        cache_path = os.path.join(self.cache_dir, f"{prefix}_interaction_features.npy")

        if load_cache and os.path.exists(cache_path):
            logger.info(f"Loading cached {prefix} features from {cache_path}")
            return np.load(cache_path)

        logger.info(f"Computing {prefix} interaction features from scratch...")
        feature_list = []
        model_names = sorted(model_preds.keys())  # Ensure deterministic order

        for name in model_names:
            preds = model_preds[name]
            probs = preds["probs"]
            pred_fs = preds["fsizes"]

            # Generate interaction feature
            feat = create_interaction_features(probs, pred_fs, true_fsizes)
            feature_list.append(feat)

        # Concatenate along columns
        X = np.hstack(feature_list)

        # Save to cache
        np.save(cache_path, X)
        logger.info(f"Saved {prefix} features to {cache_path}")

        return X


class StackingPipeline:
    """
    Orchestrates the stacking process.
    """

    def __init__(self):
        self.data_manager = StackingDataManager()
        self.meta_learner = MetaLearner()

    def run(self, oof_predictions, test_predictions, load_cache=True):
        """
        Executes the stacking pipeline.

        Args:
            oof_predictions (dict): {model_name: {'probs': array, 'fsizes': array}}
            test_predictions (dict): {model_name: {'probs': array, 'fsizes': array}}
            load_cache (bool): Use caching for feature generation.
        """
        seed_everything()

        # 1. Load Ground Truth
        gt_data = self.data_manager.load_ground_truth()
        y_train = gt_data["train_labels"]
        train_fs_true = gt_data["train_fsizes"]
        test_fs_true = gt_data["test_fsizes"]
        test_ids = gt_data["test_ids"]

        # 2. Prepare Features
        # Check alignment (basic check)
        first_model = next(iter(oof_predictions.values()))
        if len(first_model["probs"]) != len(y_train):
            logger.warning(
                f"OOF length {len(first_model['probs'])} != GT length {len(y_train)}. "
                "Assuming OOF provided matches the provided GT subset. Truncating/Aligning not performed."
            )

        X_train = self.data_manager.get_features(
            oof_predictions, train_fs_true, prefix="train", load_cache=load_cache
        )

        X_test = self.data_manager.get_features(
            test_predictions, test_fs_true, prefix="test", load_cache=load_cache
        )

        # 3. Train Meta-Learner
        self.meta_learner.fit(X_train, y_train)

        # 4. Predict on Test
        final_probs = self.meta_learner.predict(X_test)

        # 5. Save Submission
        self._save_submission(test_ids, final_probs)

        return final_probs

    def _save_submission(self, ids, probs):
        """
        Saves the submission file.
        """
        submission_path = Config.SUBMISSION_PATH
        logger.info(f"Saving submission to {submission_path}")

        df = pd.DataFrame({"id": ids, "has_cactus": probs})

        # Ensure directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        df.to_csv(submission_path, index=False)
        logger.info("Submission saved successfully.")


# Helper function to run the pipeline if called externally
def run_stacking(oof_preds, test_preds, load_cache=True):
    pipeline = StackingPipeline()
    return pipeline.run(oof_preds, test_preds, load_cache=load_cache)
