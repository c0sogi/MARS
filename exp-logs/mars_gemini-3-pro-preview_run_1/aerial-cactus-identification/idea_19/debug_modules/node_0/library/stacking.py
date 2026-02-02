import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from library.config import Config
from library.utils import calculate_roc_auc


def get_data_vectors(metadata_path, cache_prefix, load_cached_data=True):
    """
    Retrieves metadata vectors (ids, labels, file_sizes) efficiently.
    Attempts to load from cache first, otherwise computes from metadata CSV.
    Does NOT load images to save memory/time.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, labels, file_sizes) as numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    fs_cache_path = os.path.join(cache_dir, f"{cache_prefix}_fsizes.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(lbl_cache_path)
            and os.path.exists(fs_cache_path)
            and os.path.exists(id_cache_path)
        ):
            # print(f"Loading {cache_prefix} vectors from cache...")
            labels = np.load(lbl_cache_path)
            file_sizes = np.load(fs_cache_path)
            ids = np.load(id_cache_path, allow_pickle=True)
            return ids, labels, file_sizes

    # 2. Process from scratch (Metadata only)
    # print(f"Processing {cache_prefix} vectors from scratch...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    ids = df["id"].values
    # Handle missing labels (e.g. for test set placeholder is 0.5)
    labels = df["has_cactus"].values.astype(np.float32)

    # Compute file sizes
    # Note: file_path in metadata is relative to input dir
    file_paths = (
        df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x)).tolist()
    )
    fs_list = []

    for fpath in file_paths:
        if os.path.exists(fpath):
            fs_list.append(os.path.getsize(fpath))
        else:
            # Should not happen given metadata validation, but handle gracefully
            fs_list.append(0)

    file_sizes = np.array(fs_list, dtype=np.float32)

    # 3. Save to cache
    # We save these vectors so subsequent runs of stacking are fast.
    # Note: dataset.py saves images too. This partial save is safe as dataset.py
    # checks for image cache existence before loading.
    np.save(lbl_cache_path, labels)
    np.save(fs_cache_path, file_sizes)
    np.save(id_cache_path, ids)

    return ids, labels, file_sizes


class StackingEnsemble:
    """
    Implements the Heterogeneous Chromatic-Quality Stacking Ensemble.
    Uses Logistic Regression on specialist predictions + file size metadata.
    """

    def __init__(self):
        # Pipeline: Scale features -> Logistic Regression
        # StandardScaler is crucial because predictions are [0,1] but file sizes are large integers.
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0, solver="liblinear", random_state=Config.SEED
                    ),
                ),
            ]
        )

    def _create_feature_matrix(self, preds_dict, file_sizes):
        """
        Concatenates predictions and file size into a feature matrix.
        Ensures deterministic order of models.
        """
        # Sort model names to ensure consistent column ordering across train/test
        model_names = sorted(preds_dict.keys())

        features = []
        for name in model_names:
            preds = preds_dict[name]
            # Ensure shape (N, 1)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            features.append(preds)

        # Add file size as feature
        if file_sizes.ndim == 1:
            file_sizes = file_sizes.reshape(-1, 1)
        features.append(file_sizes)

        # Stack horizontally: [Pred1, Pred2, ..., PredN, FileSize]
        X = np.hstack(features)
        return X

    def fit(self, preds_dict, labels, file_sizes):
        """
        Trains the logistic regression meta-learner.
        """
        X = self._create_feature_matrix(preds_dict, file_sizes)
        self.model.fit(X, labels)

        # Calculate training score for monitoring
        probs = self.model.predict_proba(X)[:, 1]
        auc = calculate_roc_auc(labels, probs)
        return auc

    def predict(self, preds_dict, file_sizes):
        """
        Generates probability predictions.
        """
        X = self._create_feature_matrix(preds_dict, file_sizes)
        probs = self.model.predict_proba(X)[:, 1]
        return probs


def train_meta_learner(oof_preds_dict, labels, file_sizes, save_path=None):
    """
    Trains the meta-learner on OOF predictions and metadata.

    Args:
        oof_preds_dict (dict): Dictionary mapping model names to OOF probability arrays.
        labels (np.array): Ground truth labels.
        file_sizes (np.array): Image file sizes.
        save_path (str, optional): Path to save the trained model.

    Returns:
        model: Trained StackingEnsemble instance.
        score: ROC AUC score on the training data.
    """
    ensemble = StackingEnsemble()
    score = ensemble.fit(oof_preds_dict, labels, file_sizes)

    print(f"Meta-Learner Training AUC: {score:.16f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(ensemble, f)

    return ensemble, score


def predict_stacked(
    model, test_preds_dict, test_file_sizes, test_ids, output_path=None
):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        model (StackingEnsemble): Trained meta-learner.
        test_preds_dict (dict): Dictionary mapping model names to Test probability arrays.
        test_file_sizes (np.array): Test image file sizes.
        test_ids (np.array): Test image IDs.
        output_path (str, optional): Path to save submission CSV.

    Returns:
        df: Pandas DataFrame with predictions.
    """
    probs = model.predict(test_preds_dict, test_file_sizes)

    df = pd.DataFrame({"id": test_ids, "has_cactus": probs})

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
