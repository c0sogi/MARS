import os
import hashlib
import numpy as np
import torch
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import get_score


def get_cache_filename(mode: str, tta: bool) -> str:
    """
    Generates a deterministic cache filename based on configuration.
    """
    # Create a unique hash based on critical configuration parameters
    # Added new backbones to hash to ensure cache invalidation
    config_str = f"{mode}_{tta}_{Config.IMG_SIZE}_{Config.BACKBONE_SWIN}_{Config.BACKBONE_EFFNET}_{Config.BACKBONE_DINO}_{Config.BACKBONE_CLIP}"
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
    1. SVR (RBF Kernel) with PCA (Tuned C=20.0)
    2. LightGBM
    3. ExtraTrees Regressor (Replaces Ridge)

    Level 2 Meta-Learner:
    - Linear Regression
    """

    def __init__(self, backbone_dims, seed: int = 42):
        self.seed = seed
        self.backbone_dims = backbone_dims
        self.fitted = False
        self.base_models = {}
        self.meta_model = LinearRegression()

        # Preprocessing pipelines for each backbone (Independent PCA)
        self.feature_pipelines = []

    def _get_feature_slices(self):
        """Helper to get slice indices for each backbone + metadata"""
        slices = []
        start = 0
        for dim in self.backbone_dims:
            slices.append(slice(start, start + dim))
            start += dim
        # Metadata slice (last 12 columns)
        slices.append(slice(start, None))
        return slices

    def _fit_transform_features(self, X, fit=True):
        """
        Applies Independent PCA to each backbone stream and scales metadata.
        Cite solution_lesson_node_00022 (Independent Dimensionality Reduction).
        """
        slices = self._get_feature_slices()
        transformed_parts = []

        # Process backbone features
        for i in range(len(self.backbone_dims)):
            part = X[:, slices[i]]

            if fit:
                # Initialize pipeline if fitting
                # Retain 95% variance
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=0.95, random_state=self.seed)),
                    ]
                )
                if len(self.feature_pipelines) <= i:
                    self.feature_pipelines.append(pipe)
                else:
                    self.feature_pipelines[i] = pipe

                trans = pipe.fit_transform(part)
            else:
                trans = self.feature_pipelines[i].transform(part)

            transformed_parts.append(trans)

        # Process Metadata (Scale by 10.0) - Cite solution_lesson_node_00025
        meta_part = X[:, slices[-1]] * 10.0
        transformed_parts.append(meta_part)

        return np.concatenate(transformed_parts, axis=1)

    def _create_base_models(self):
        """
        Instantiates fresh base models with fixed seeds.
        """
        # 1. SVR: C=20.0 (Cite solution_lesson_node_00026)
        # Note: PCA is now handled globally before this, so we just scale here
        svr_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(C=20.0, epsilon=0.1, kernel="rbf")),
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
            n_jobs=1,
            verbose=-1,
        )

        # 3. ExtraTrees (Replaces Ridge) - Cite solution_lesson_node_00026
        # Disable feature subsampling (max_features=None) - Cite solution_lesson_node_00027
        et_model = ExtraTreesRegressor(
            n_estimators=200,
            max_features=None,
            min_samples_leaf=5,
            random_state=self.seed,
            n_jobs=1,
        )

        return {"svr": svr_model, "lgbm": lgbm_model, "et": et_model}

    def cross_validate_and_fit(self, X: np.ndarray, y: np.ndarray):
        print("Initializing Stacking Cross-Validation...")
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=self.seed)

        model_keys = ["svr", "lgbm", "et"]
        oof_preds = np.zeros((X.shape[0], len(model_keys)))

        # Iterate through folds
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Independent PCA Fit/Transform on Train, Transform on Val
            # We reset pipelines for each fold to avoid leakage
            self.feature_pipelines = []
            X_train = self._fit_transform_features(X_train_raw, fit=True)
            X_val = self._fit_transform_features(X_val_raw, fit=False)

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

        final_oof_preds = self.meta_model.predict(oof_preds)
        final_score = get_score(y, final_oof_preds)
        print(f"Stacking Ensemble OOF RMSE: {final_score}")

        # Retrain on Full Dataset
        print("Retraining Base Models on full dataset...")
        # Fit PCA on full dataset
        self.feature_pipelines = []
        X_full = self._fit_transform_features(X, fit=True)

        self.base_models = self._create_base_models()
        for key in model_keys:
            self.base_models[key].fit(X_full, y)

        self.fitted = True
        return final_score

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Run cross_validate_and_fit first.")

        # Transform Test features using pipelines fitted on full train set
        X_test_trans = self._fit_transform_features(X_test, fit=False)

        model_keys = ["svr", "lgbm", "et"]
        test_preds_matrix = np.zeros((X_test_trans.shape[0], len(model_keys)))

        for i, key in enumerate(model_keys):
            model = self.base_models[key]
            test_preds_matrix[:, i] = model.predict(X_test_trans)

        final_preds = self.meta_model.predict(test_preds_matrix)
        final_preds = np.clip(final_preds, 1.0, 100.0)

        return final_preds
