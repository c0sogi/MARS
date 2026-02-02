import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything
from library.architectures import WhaleClassifier
from library.data_factory import WhaleDataset


class Predictor:
    """
    Handles the inference pipeline for the Right Whale Detection task.
    Manages model loading, OOF prediction generation, Meta-Learner training,
    and final test set submission generation.
    """

    def __init__(self, debug=Config.DEBUG):
        """
        Args:
            debug (bool): If True, runs on a subset of data for testing.
        """
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(Config.SEED)

        # Ensure directories exist
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    def _load_model(self, arch, checkpoint_path):
        """
        Loads a specific model architecture and weights.

        Args:
            arch (str): Architecture name (e.g., 'tf_efficientnet_b0_ns').
            checkpoint_path (str): Path to the .pth file.

        Returns:
            model (nn.Module) or None: Loaded model in eval mode, or None if missing.
        """
        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint not found at {checkpoint_path}. Using neutral predictions."
            )
            return None

        try:
            # Instantiate model with pretrained=False since we load custom weights
            model = WhaleClassifier(arch, pretrained=False)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None

    def _predict_loader(self, model, loader):
        """
        Generates predictions for a DataLoader using a single model.

        Returns:
            np.array: Flattened array of probabilities.
        """
        preds = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                # Apply Sigmoid to get probabilities [0, 1]
                probs = torch.sigmoid(outputs)
                preds.append(probs.cpu().numpy())

        if len(preds) == 0:
            return np.array([])
        return np.concatenate(preds).flatten()

    def generate_oof_predictions(self, load_cached_data=True):
        """
        Generates Out-Of-Fold predictions for all ensemble configurations.
        These serve as features for the Meta-Learner.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X_oof, y_oof)
                X_oof (np.array): Features of shape (N_samples, N_configs)
                y_oof (np.array): Ground truth labels
        """
        cache_x_path = os.path.join(Config.CACHE_DIR, "oof_features.npy")
        cache_y_path = os.path.join(Config.CACHE_DIR, "oof_targets.npy")

        if (
            load_cached_data
            and os.path.exists(cache_x_path)
            and os.path.exists(cache_y_path)
        ):
            print("Loading cached OOF features...")
            return np.load(cache_x_path), np.load(cache_y_path)

        print("Generating OOF features from scratch (Train Only)...")

        # Load ONLY Train metadata
        full_df = pd.read_csv(Config.TRAIN_CSV)

        if self.debug:
            full_df = full_df.sample(
                n=min(len(full_df), 200), random_state=Config.SEED
            ).reset_index(drop=True)

        # Initialize storage
        n_samples = len(full_df)
        n_configs = len(Config.ENSEMBLE_CONFIGS)
        oof_preds = np.zeros((n_samples, n_configs))

        # Reconstruct Stratified K-Fold Split
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        for fold, (_, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
            print(f"Processing OOF Fold {fold}/{Config.N_FOLDS - 1}...")

            # Prepare Validation Data for this Fold
            fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)
            dataset = WhaleDataset(fold_val_df, phase="val")
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Iterate over Ensemble Configurations
            for config_idx, config in enumerate(Config.ENSEMBLE_CONFIGS):
                # Construct expected checkpoint path: {name}_fold_{fold}.pth
                ckpt_name = f"{config['name']}_fold_{fold}.pth"
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)

                model = self._load_model(config["arch"], ckpt_path)

                if model is not None:
                    preds = self._predict_loader(model, loader)
                    # Store predictions in the corresponding indices
                    oof_preds[val_idx, config_idx] = preds
                else:
                    # Fallback for missing models
                    oof_preds[val_idx, config_idx] = 0.5

        y_oof = full_df["label"].values

        # Cache results
        np.save(cache_x_path, oof_preds)
        np.save(cache_y_path, y_oof)

        return oof_preds, y_oof

    def generate_bagged_features(self, df, cache_name=None):
        """
        Generates features for a generic dataframe using Bagging.
        Averages predictions from all 5 folds for each configuration.
        Used for Validation (Hold-out) and Test sets.
        """
        if cache_name:
            cache_path = os.path.join(Config.CACHE_DIR, cache_name)
            if os.path.exists(cache_path):
                print(f"Loading cached features from {cache_name}...")
                return np.load(cache_path)

        print(f"Generating Bagged features for {len(df)} samples...")

        if self.debug:
            df = df.sample(n=min(len(df), 100), random_state=Config.SEED).reset_index(
                drop=True
            )

        dataset = WhaleDataset(
            df, phase="test"
        )  # Use 'test' phase to avoid augmentation
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        n_configs = len(Config.ENSEMBLE_CONFIGS)
        bagged_preds = np.zeros((len(df), n_configs))

        for config_idx, config in enumerate(Config.ENSEMBLE_CONFIGS):
            # print(f"Processing Config: {config['name']}...")
            fold_preds_list = []

            # Aggregate predictions from all available folds
            for fold in range(Config.N_FOLDS):
                ckpt_name = f"{config['name']}_fold_{fold}.pth"
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)

                model = self._load_model(config["arch"], ckpt_path)
                if model is not None:
                    preds = self._predict_loader(model, loader)
                    fold_preds_list.append(preds)

            if fold_preds_list:
                # Average across folds (Bagging)
                avg_preds = np.mean(fold_preds_list, axis=0)
                bagged_preds[:, config_idx] = avg_preds
            else:
                bagged_preds[:, config_idx] = 0.5

        if cache_name:
            np.save(cache_path, bagged_preds)

        return bagged_preds

    def generate_test_predictions(self, load_cached_data=True):
        """
        Wrapper for Test set prediction generation.
        """
        test_df = pd.read_csv(Config.TEST_CSV)
        return self.generate_bagged_features(test_df, "test_features.npy")

    def create_submission(self, meta_learner_override=None):
        """
        Executes the submission generation.
        Args:
            meta_learner_override: If provided, uses this trained model instead of retraining.
        """
        print("Starting Submission Pipeline...")

        if meta_learner_override:
            meta_learner = meta_learner_override
        else:
            # Retrain if not provided (fallback)
            X_oof, y_oof = self.generate_oof_predictions(load_cached_data=True)
            X_oof = np.nan_to_num(X_oof, nan=0.5)
            print("Training Meta-Learner (Logistic Regression)...")
            meta_learner = LogisticRegression(
                random_state=Config.SEED, solver="liblinear"
            )
            meta_learner.fit(X_oof, y_oof)

        # Prepare Test Data
        X_test = self.generate_test_predictions(load_cached_data=True)
        X_test = np.nan_to_num(X_test, nan=0.5)

        # Generate Final Predictions
        final_probs = meta_learner.predict_proba(X_test)[:, 1]

        # Create Submission DataFrame
        test_df = pd.read_csv(Config.TEST_CSV)
        if self.debug:
            test_df = test_df.sample(
                n=min(len(test_df), 100), random_state=Config.SEED
            ).reset_index(drop=True)

        submission = pd.DataFrame({"clip": test_df["clip"], "probability": final_probs})

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(save_path, index=False)
        print(f"Submission saved successfully to {save_path}")
        print(submission.head())
