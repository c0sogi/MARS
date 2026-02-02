import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library.config import (
    MODEL_ARCHITECTURES,
    NUM_FOLDS,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CHECKPOINT_DIR,
    CACHE_DIR,
    SUBMISSION_FILE,
    DEVICE,
    BASE_OUTPUT_DIR,
)
from library.dataset import load_and_cache_data, CactusDataset, get_transforms
from library.inference import generate_fold_predictions
from library.utils import get_logger

# Initialize Logger
logger = get_logger("stacking")


def get_full_train_data(load_cached_data=True):
    """
    Combines train and validation metadata to create the full dataset for CV.
    """
    # Load Train Part
    ids_t, imgs_t, lbls_t = load_and_cache_data(
        TRAIN_META_PATH, "train_stacking", load_cached_data=load_cached_data
    )
    # Load Val Part
    ids_v, imgs_v, lbls_v = load_and_cache_data(
        VAL_META_PATH, "val_stacking", load_cached_data=load_cached_data
    )

    # Concatenate
    ids = np.concatenate([ids_t, ids_v])
    images = np.concatenate([imgs_t, imgs_v])
    labels = np.concatenate([lbls_t, lbls_v])

    return ids, images, labels


def get_test_data(load_cached_data=True):
    """
    Loads test data.
    """
    ids, images, labels = load_and_cache_data(
        TEST_META_PATH, "test_stacking", load_cached_data=load_cached_data
    )
    return ids, images, labels


def prepare_meta_features(load_cached_data=True):
    """
    Generates or loads the meta-features for stacking.
    Constructs X_train (OOF) and X_test (Averaged).
    """
    cache_path = os.path.join(CACHE_DIR, "stacking_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading stacked features from {cache_path}")
        try:
            full_df = pd.read_parquet(cache_path)
            # Split back into train (OOF) and test
            train_df = full_df[full_df["is_test"] == 0].reset_index(drop=True)
            test_df = full_df[full_df["is_test"] == 1].reset_index(drop=True)
            return train_df, test_df
        except Exception as e:
            logger.warning(f"Failed to load stacking cache: {e}. Recomputing...")

    logger.info("Generating meta-features from base models...")

    # 1. Load Data
    train_ids, train_imgs, train_lbls = get_full_train_data(load_cached_data)
    test_ids, test_imgs, test_lbls = get_test_data(load_cached_data)

    # 2. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Prepare containers
    # We need to store OOF predictions for every training sample
    # Structure: id, target, model1_mean, model1_std, model2_mean...
    oof_data = {"id": train_ids, "target": train_lbls}

    # For test, we will collect predictions from all folds and average them later
    # Structure: model_name -> list of dataframes (one per fold)
    test_preds_collection = {model: [] for model in MODEL_ARCHITECTURES}

    # Initialize columns in oof_data
    for model_name in MODEL_ARCHITECTURES:
        oof_data[f"{model_name}_mean"] = np.zeros_like(train_lbls)
        oof_data[f"{model_name}_std"] = np.zeros_like(train_lbls)

    # 3. Iterate Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_imgs, train_lbls)):
        logger.info(f"Processing Fold {fold}/{NUM_FOLDS - 1}...")

        # Create Loaders for this fold
        # Val Loader (for OOF)
        val_ds = CactusDataset(
            train_ids[val_idx],
            train_imgs[val_idx],
            train_lbls[val_idx],
            transform=get_transforms("val"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Test Loader (re-used per fold to get predictions from that fold's model)
        test_ds = CactusDataset(
            test_ids, test_imgs, test_lbls, transform=get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Iterate Models
        for model_name in MODEL_ARCHITECTURES:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"{model_name}_fold{fold}.pth"
            )

            # Generate predictions (handles caching internally per model/fold)
            val_pred_df, test_pred_df = generate_fold_predictions(
                model_name=model_name,
                fold_idx=fold,
                checkpoint_path=checkpoint_path,
                val_loader=val_loader,
                test_loader=test_loader,
                device=DEVICE,
                load_cached_data=load_cached_data,
            )

            # Map OOF predictions back to the main array
            # We use a mapping dictionary for O(1) lookup or sorting
            # Since val_pred_df['id'] matches train_ids[val_idx], we can align strictly

            # Create a map from ID to prediction
            val_mean_map = dict(zip(val_pred_df["id"], val_pred_df["pred_mean"]))
            val_std_map = dict(zip(val_pred_df["id"], val_pred_df["pred_std"]))

            # Assign to the correct indices in the full arrays
            current_val_ids = train_ids[val_idx]
            for i, vid in enumerate(current_val_ids):
                global_idx = val_idx[i]
                oof_data[f"{model_name}_mean"][global_idx] = val_mean_map.get(vid, 0.5)
                oof_data[f"{model_name}_std"][global_idx] = val_std_map.get(vid, 0.0)

            # Collect Test predictions
            test_preds_collection[model_name].append(test_pred_df)

    # 4. Construct DataFrames
    train_df = pd.DataFrame(oof_data)
    train_df["is_test"] = 0

    # Process Test Data (Average across folds)
    test_data = {"id": test_ids}
    # Target is dummy for test, but we keep structure
    test_data["target"] = test_lbls

    for model_name in MODEL_ARCHITECTURES:
        # Concatenate all folds: List of 5 DFs
        fold_dfs = test_preds_collection[model_name]

        # We assume rows are aligned because we used the same loader/order
        # But to be safe, let's concat and groupby ID
        combined_test = pd.concat(fold_dfs)
        averaged_test = (
            combined_test.groupby("id")[["pred_mean", "pred_std"]].mean().reset_index()
        )

        # Ensure alignment with test_ids
        # Create map
        mean_map = dict(zip(averaged_test["id"], averaged_test["pred_mean"]))
        std_map = dict(zip(averaged_test["id"], averaged_test["pred_std"]))

        test_data[f"{model_name}_mean"] = [mean_map.get(tid, 0.5) for tid in test_ids]
        test_data[f"{model_name}_std"] = [std_map.get(tid, 0.0) for tid in test_ids]

    test_df = pd.DataFrame(test_data)
    test_df["is_test"] = 1

    # 5. Save to Cache
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)

    return train_df, test_df


class MetaLearner:
    def __init__(self):
        self.model = LogisticRegression(solver="liblinear", random_state=SEED)
        self.features = []
        for arch in MODEL_ARCHITECTURES:
            self.features.append(f"{arch}_mean")
            self.features.append(f"{arch}_std")

    def fit(self, df):
        X = df[self.features].values
        y = df["target"].values
        self.model.fit(X, y)

        # Evaluate on training data (OOF)
        preds = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)
        logger.info(f"Meta-Learner OOF AUC: {auc:.8f}")

        # Log coefficients
        logger.info("Meta-Learner Coefficients:")
        for name, coef in zip(self.features, self.model.coef_[0]):
            logger.info(f"  {name}: {coef:.4f}")

    def predict(self, df):
        X = df[self.features].values
        return self.model.predict_proba(X)[:, 1]


def train_stacking_model(load_cached_data=True):
    """
    Main function to run the stacking pipeline.
    """
    logger.info("Starting Stacking Pipeline...")

    # 1. Prepare Data
    train_df, test_df = prepare_meta_features(load_cached_data=load_cached_data)

    # 2. Train Meta-Learner
    meta_model = MetaLearner()
    meta_model.fit(train_df)

    # 3. Predict on Test
    test_probs = meta_model.predict(test_df)

    # 4. Create Submission
    submission = pd.DataFrame({"id": test_df["id"], "has_cactus": test_probs})

    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
    submission.to_csv(SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {SUBMISSION_FILE}")

    # Save Meta Model
    model_path = os.path.join(BASE_OUTPUT_DIR, "meta_model.joblib")
    joblib.dump(meta_model.model, model_path)
    logger.info(f"Meta-model saved to {model_path}")
