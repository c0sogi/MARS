import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from library.config import Config
from library.utils import get_logger, calculate_log_loss

# Initialize logger
logger = get_logger("stacking")


class StackingTrainer:
    """
    Manages the training of the meta-learner for the stacking ensemble.
    """

    def __init__(
        self,
        model_names=Config.MODEL_NAMES,
        oof_dir=None,
        output_dir=None,
    ):
        self.model_names = model_names
        self.oof_dir = oof_dir if oof_dir is not None else Config.OOF_DIR
        self.output_dir = (
            output_dir if output_dir is not None else Config.CHECKPOINT_DIR
        )
        self.meta_model_path = os.path.join(self.output_dir, "meta_model.joblib")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_oof_data(self, load_cached_data=True):
        """
        Loads and aggregates Out-Of-Fold predictions from base models.
        Implements caching to parquet to avoid re-processing CSVs.
        """
        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "meta_oof_data.parquet")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_file):
            logger.info(f"Loading cached OOF data from {cache_file}")
            try:
                meta_df = pd.read_parquet(cache_file)
                # Verify all model columns exist
                if all(m in meta_df.columns for m in self.model_names):
                    X = meta_df[self.model_names].values
                    y = meta_df["label"].values
                    return X, y
            except Exception as e:
                logger.info(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        logger.info("Constructing OOF dataset from base model predictions...")

        # Load ground truth
        if not os.path.exists(Config.TRAIN_CSV):
            raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_CSV}")

        base_df = pd.read_csv(Config.TRAIN_CSV)
        # We use filepath as the key to merge
        meta_df = base_df[["filepath", "label"]].copy()

        for model_name in self.model_names:
            # Check for consolidated OOF file
            consolidated_path = os.path.join(self.oof_dir, f"{model_name}_oof.csv")

            if os.path.exists(consolidated_path):
                df_oof = pd.read_csv(consolidated_path)
                # Expect columns: filepath, pred (or similar)
                if "pred" in df_oof.columns:
                    df_oof = df_oof.rename(columns={"pred": model_name})

                # Merge
                if "filepath" not in df_oof.columns:
                    # If no filepath, assume index alignment if lengths match
                    if len(df_oof) == len(meta_df):
                        meta_df[model_name] = df_oof[model_name]
                    else:
                        raise ValueError(
                            f"OOF file for {model_name} missing 'filepath' column and length mismatch."
                        )
                else:
                    meta_df = meta_df.merge(
                        df_oof[["filepath", model_name]], on="filepath", how="left"
                    )

            else:
                # Check for per-fold files
                fold_dfs = []
                for fold in range(Config.N_FOLDS):
                    fold_path = os.path.join(
                        self.oof_dir, f"{model_name}_fold_{fold}_oof.csv"
                    )
                    if os.path.exists(fold_path):
                        df_fold = pd.read_csv(fold_path)
                        if "pred" in df_fold.columns:
                            df_fold = df_fold.rename(columns={"pred": model_name})
                        fold_dfs.append(df_fold)

                if not fold_dfs:
                    raise FileNotFoundError(
                        f"No OOF files found for model {model_name} in {self.oof_dir}"
                    )

                # Concatenate all folds
                full_oof = pd.concat(fold_dfs, ignore_index=True)

                # Merge
                if "filepath" in full_oof.columns:
                    meta_df = meta_df.merge(
                        full_oof[["filepath", model_name]], on="filepath", how="left"
                    )
                else:
                    raise ValueError(
                        f"Per-fold OOF files for {model_name} must contain 'filepath' for alignment."
                    )

            # Check for NaNs after merge
            if meta_df[model_name].isnull().any():
                logger.info(
                    f"Warning: {meta_df[model_name].isnull().sum()} missing predictions for {model_name}. Filling with 0.5."
                )
                meta_df[model_name] = meta_df[model_name].fillna(0.5)

        # Save to cache
        meta_df.to_parquet(cache_file)

        X = meta_df[self.model_names].values
        y = meta_df["label"].values

        return X, y

    def train(self, load_cached_data=True):
        """
        Trains the meta-learner.
        """
        X, y = self.load_oof_data(load_cached_data=load_cached_data)

        logger.info(
            f"Training Meta-Learner on {len(X)} samples with features: {self.model_names}"
        )

        # Initialize Meta Learner (Logistic Regression)
        clf = LogisticRegression(random_state=Config.SEED, solver="liblinear", C=1.0)

        clf.fit(X, y)

        # Evaluate on OOF (sanity check)
        preds = clf.predict_proba(X)[:, 1]
        loss = calculate_log_loss(y, preds)

        logger.info(f"Meta-Learner OOF Log Loss: {loss:.15f}")
        logger.info(f"Model Coefficients: {dict(zip(self.model_names, clf.coef_[0]))}")
        logger.info(f"Intercept: {clf.intercept_[0]}")

        # Save model
        joblib.dump(clf, self.meta_model_path)
        logger.info(f"Meta-learner saved to {self.meta_model_path}")

        return clf


def predict_stacking(load_cached_data=True):
    """
    Generates predictions for the test set using the trained meta-learner.
    Aggregates base model predictions (averaging across folds) and then applies the meta-model.
    """
    trainer = StackingTrainer()

    # 1. Load Meta Model
    if not os.path.exists(trainer.meta_model_path):
        raise FileNotFoundError(
            f"Meta model not found at {trainer.meta_model_path}. Please train first."
        )

    clf = joblib.load(trainer.meta_model_path)
    logger.info(f"Loaded Meta-Learner from {trainer.meta_model_path}")

    # 2. Prepare Test Data (Meta Features)
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "meta_test_data.parquet")

    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading cached Test Meta data from {cache_file}")
        test_meta_df = pd.read_parquet(cache_file)
    else:
        logger.info("Constructing Test Meta data from base model predictions...")

        if not os.path.exists(Config.TEST_CSV):
            raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

        test_df = pd.read_csv(Config.TEST_CSV)
        test_meta_df = test_df[["id"]].copy()

        for model_name in Config.MODEL_NAMES:
            # We need to aggregate predictions from 5 folds for this model
            fold_preds = []

            # Look for per-fold test predictions
            for fold in range(Config.N_FOLDS):
                pred_path = os.path.join(
                    Config.OOF_DIR, f"{model_name}_fold_{fold}_test.csv"
                )
                if os.path.exists(pred_path):
                    df_fold = pd.read_csv(pred_path)
                    if "pred" in df_fold.columns:
                        # Rename to a temporary name to avoid collision during merge
                        df_fold = df_fold.rename(columns={"pred": f"pred_{fold}"})
                    fold_preds.append(df_fold[["id", f"pred_{fold}"]])

            if not fold_preds:
                # Fallback: check for a single pre-averaged file
                single_path = os.path.join(Config.OOF_DIR, f"{model_name}_test.csv")
                if os.path.exists(single_path):
                    df_single = pd.read_csv(single_path)
                    if "pred" in df_single.columns:
                        test_meta_df = test_meta_df.merge(
                            df_single[["id", "pred"]].rename(
                                columns={"pred": model_name}
                            ),
                            on="id",
                            how="left",
                        )
                    continue
                else:
                    raise FileNotFoundError(
                        f"No test predictions found for {model_name}"
                    )

            # Merge all folds for this model
            merged_model = fold_preds[0]
            for i in range(1, len(fold_preds)):
                merged_model = merged_model.merge(fold_preds[i], on="id", how="inner")

            # Calculate mean across folds
            pred_cols = [c for c in merged_model.columns if c.startswith("pred_")]
            merged_model[model_name] = merged_model[pred_cols].mean(axis=1)

            # Merge into main meta dataframe
            test_meta_df = test_meta_df.merge(
                merged_model[["id", model_name]], on="id", how="left"
            )

        # Cache
        test_meta_df.to_parquet(cache_file)

    # 3. Predict
    logger.info(f"Generating predictions for {len(test_meta_df)} test samples...")
    X_test = test_meta_df[Config.MODEL_NAMES].values

    final_probs = clf.predict_proba(X_test)[:, 1]

    # 4. Create Submission
    submission = pd.DataFrame({"id": test_meta_df["id"], "label": final_probs})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
