import os
import hashlib
import numpy as np
import torch
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import get_score


def get_cache_filename(mode: str, tta: bool) -> str:
    """
    Generates a deterministic cache filename based on configuration.
    """
    # Create a unique hash based on critical configuration parameters
    config_str = f"{mode}_{tta}_{Config.IMG_SIZE}_{Config.BACKBONE_SWIN}_{Config.BACKBONE_EFFNET}"
    hash_str = hashlib.md5(config_str.encode()).hexdigest()
    filename = f"features_{mode}_{hash_str}.npy"
    return os.path.join(Config.WORKING_DIR, filename)


def extract_features_and_cache(
    model,
    dataloader,
    device,
    mode: str,
    tta: bool = False,
    load_cached_data: bool = True,
):
    """
    Extracts features from the model using the provided dataloader.
    Implements caching to disk to avoid re-computation.

    Args:
        model: The PyTorch model (AdaptiveBackbone).
        dataloader: The PyTorch DataLoader.
        device: Torch device.
        mode: 'train', 'valid', or 'test'.
        tta: Boolean, whether to use Test-Time Augmentation (Horizontal Flip).
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        tuple: (features, targets, ids)
    """
    cache_path = get_cache_filename(mode, tta)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data["features"], data["targets"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing features...")

    # 2. Compute features
    print(f"Extracting features for mode='{mode}' with TTA={tta}...")
    model.eval()

    features_list = []
    targets_list = []
    ids_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch["image"].to(device)
            dense = batch["dense_features"].to(device)
            ids = batch["id"]

            # Forward pass (Original)
            # Returns embeddings: (B, embedding_dim)
            emb = model(images, feature_extract=True)

            # Test-Time Augmentation (Horizontal Flip)
            if tta:
                # Flip images horizontally (dim 3 is width in NCHW)
                images_flip = torch.flip(images, dims=[3])
                emb_flip = model(images_flip, feature_extract=True)
                # Average embeddings
                emb = (emb + emb_flip) / 2.0

            # Concatenate Image Embeddings + Dense Metadata
            # Dense shape: (B, 12)
            # Result shape: (B, embedding_dim + 12)
            full_features = torch.cat([emb, dense], dim=1)

            features_list.append(full_features.cpu().numpy())
            ids_list.extend(ids)

            # Handle targets if present
            if "target" in batch:
                targets_list.append(batch["target"].cpu().numpy())

    # Concatenate all batches
    features = np.concatenate(features_list, axis=0)
    ids = np.array(ids_list)

    if targets_list:
        targets = np.concatenate(targets_list, axis=0)
    else:
        targets = None

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_data = {"features": features, "targets": targets, "ids": ids}
    np.save(cache_path, cache_data)
    print(f"Features saved to {cache_path}")

    return features, targets, ids


class StackingRegressor:
    """
    Stage 2: Heterogeneous Stacking Ensemble.

    Level 1 Base Learners:
    1. SVR (RBF Kernel) with PCA
    2. LightGBM
    3. Ridge Regression

    Level 2 Meta-Learner:
    - Linear Regression
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.fitted = False
        self.base_models = {}
        self.meta_model = LinearRegression()

    def _create_base_models(self):
        """
        Instantiates fresh base models with fixed seeds.
        """
        # 1. SVR Pipeline: Scaling -> PCA -> SVR
        # PCA reduces dimensionality for the kernel method to improve speed and generalization
        svr_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=64, random_state=self.seed)),
                ("svr", SVR(C=1.0, epsilon=0.1, kernel="rbf")),
            ]
        )

        # 2. LightGBM
        lgbm_model = lgb.LGBMRegressor(
            random_state=self.seed,
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,  # Parallelization handled by outer loop or restricted env
            verbose=-1,
        )

        # 3. Ridge Regression
        ridge_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=self.seed)),
            ]
        )

        return {"svr": svr_pipeline, "lgbm": lgbm_model, "ridge": ridge_pipeline}

    def cross_validate_and_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Performs K-Fold Cross-Validation to train the Meta-Learner,
        then retrains all base models on the full dataset.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.

        Returns:
            float: The OOF RMSE score of the Stacking Ensemble.
        """
        print("Initializing Stacking Cross-Validation...")
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=self.seed)

        # Storage for Out-of-Fold predictions
        # Shape: (N_samples, N_models)
        model_keys = ["svr", "lgbm", "ridge"]
        oof_preds = np.zeros((X.shape[0], len(model_keys)))

        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create fresh models for this fold
            fold_models = self._create_base_models()

            for i, key in enumerate(model_keys):
                model = fold_models[key]
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                oof_preds[val_idx, i] = pred

        # Train Meta-Learner on OOF predictions
        print("Training Meta-Learner on OOF predictions...")
        self.meta_model.fit(oof_preds, y)

        # Calculate Stacking Score
        final_oof_preds = self.meta_model.predict(oof_preds)
        final_score = get_score(y, final_oof_preds)
        print(f"Stacking Ensemble OOF RMSE: {final_score}")

        # Retrain Base Models on Full Dataset
        print("Retraining Base Models on full dataset...")
        self.base_models = self._create_base_models()
        for key in model_keys:
            self.base_models[key].fit(X, y)

        self.fitted = True
        return final_score

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the test set using the stacked ensemble.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted Pawpularity scores.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Run cross_validate_and_fit first.")

        model_keys = ["svr", "lgbm", "ridge"]
        test_preds_matrix = np.zeros((X_test.shape[0], len(model_keys)))

        # Generate predictions from base models
        for i, key in enumerate(model_keys):
            model = self.base_models[key]
            test_preds_matrix[:, i] = model.predict(X_test)

        # Aggregate using Meta-Learner
        final_preds = self.meta_model.predict(test_preds_matrix)

        # Clip predictions to valid range [1, 100]
        final_preds = np.clip(final_preds, 1.0, 100.0)

        return final_preds
