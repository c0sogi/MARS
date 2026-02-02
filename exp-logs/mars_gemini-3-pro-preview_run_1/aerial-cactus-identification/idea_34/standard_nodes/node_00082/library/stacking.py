import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.models import CactusRepVGG, CactusResNet, CactusMicroNeXt


class GeometricStacking:
    """
    Implements the Heterogeneous Geometric-Consistency Stacking Ensemble.
    Generates meta-features based on the mean and standard deviation of predictions
    across 4 geometric views (TTA), then trains a Logistic Regression meta-learner.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.models_config = Config.MODEL_ARCHITECTURES
        self.n_folds = Config.N_FOLDS

        # Paths for caching meta-features
        self.meta_features_val_path = os.path.join(
            Config.WORK_DIR, "meta_features_val.npy"
        )
        self.meta_targets_val_path = os.path.join(
            Config.WORK_DIR, "meta_targets_val.npy"
        )
        self.meta_features_test_path = os.path.join(
            Config.WORK_DIR, "meta_features_test.npy"
        )

        seed_everything(Config.SEED)

    def _load_base_model(self, model_name, fold):
        """
        Instantiates and loads a base model from checkpoint.
        Handles specific logic like RepVGG re-parameterization.
        """
        # Instantiate Architecture
        if model_name == "CactusRepVGG":
            model = CactusRepVGG(num_classes=Config.NUM_CLASSES)
        elif model_name == "CactusResNet":
            model = CactusResNet(num_classes=Config.NUM_CLASSES)
        elif model_name == "CactusMicroNeXt":
            model = CactusMicroNeXt(num_classes=Config.NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        model.to(self.device)

        # Determine Checkpoint Path (Prefer SWA, fallback to Best)
        # Note: training.py saves as f"{model_name}_swa" and f"{model_name}_best"
        # Config.get_checkpoint_path appends "_fold{fold}.pth"
        ckpt_path = Config.get_checkpoint_path(f"{model_name}_swa", fold)
        if not os.path.exists(ckpt_path):
            print(
                f"  [Warning] SWA checkpoint missing for {model_name} fold {fold}. Trying Best..."
            )
            ckpt_path = Config.get_checkpoint_path(f"{model_name}_best", fold)

        try:
            model = load_checkpoint(model, ckpt_path, self.device)
        except FileNotFoundError:
            print(
                f"  [Error] No checkpoint found for {model_name} fold {fold} at {ckpt_path}"
            )
            raise

        # Structural Re-parameterization for RepVGG
        if model_name == "CactusRepVGG":
            # Must be in eval mode or explicitly switch to deploy
            model.reparameterize()

        model.eval()
        return model

    def _predict_tta(self, model, loader):
        """
        Generates predictions using 4-view Test-Time Augmentation.
        Returns: (N, 2) array containing [Mean, Std] for each sample.
        """
        preds_list = []

        with torch.no_grad():
            for batch in loader:
                # Handle DataLoader returning (img, label) or just (img)
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)
                batch_size = images.size(0)

                # Generate 4 Geometric Views
                # 1. Original
                v1 = images
                # 2. Horizontal Flip
                v2 = torch.flip(images, [3])
                # 3. Vertical Flip
                v3 = torch.flip(images, [2])
                # 4. 180 Rotation (H + V Flip)
                v4 = torch.flip(images, [2, 3])

                # Stack views: (B, 4, C, H, W) -> Flatten to (B*4, C, H, W)
                views = torch.stack([v1, v2, v3, v4], dim=1)
                views = views.view(-1, 3, Config.IMG_SIZE, Config.IMG_SIZE)

                # Inference
                logits = model(views)
                probs = torch.sigmoid(logits)

                # Reshape back to (B, 4)
                probs = probs.view(batch_size, 4)

                # Compute Geometric Consistency Features
                mu = probs.mean(dim=1, keepdim=True)
                sigma = probs.std(dim=1, keepdim=True)

                # Concatenate [Mean, Std] -> (B, 2)
                batch_feats = torch.cat([mu, sigma], dim=1)
                preds_list.append(batch_feats.cpu().numpy())

        return np.concatenate(preds_list, axis=0)

    def generate_geometric_features(self, val_loader, test_loader, load_cached=True):
        """
        Generates or loads meta-features for Stacking.
        Iterates over all models and folds, computing TTA stats.
        """
        # 1. Try Loading from Cache
        if load_cached:
            if (
                os.path.exists(self.meta_features_val_path)
                and os.path.exists(self.meta_targets_val_path)
                and os.path.exists(self.meta_features_test_path)
            ):
                print("Loading cached geometric meta-features...")
                X_val = np.load(self.meta_features_val_path)
                y_val = np.load(self.meta_targets_val_path)
                X_test = np.load(self.meta_features_test_path)
                return X_val, y_val, X_test
            else:
                print("Cache not found. Generating features from scratch...")

        # 2. Generate from Scratch
        print(
            f"Generating features using {len(self.models_config)} architectures x {self.n_folds} folds..."
        )

        # Extract Validation Targets once
        y_val_list = []
        for _, labels in val_loader:
            y_val_list.append(labels.numpy())
        y_val = np.concatenate(y_val_list)

        val_feats_all = []
        test_feats_all = []

        # Iterate over every model instance
        for model_name in self.models_config:
            for fold in range(self.n_folds):
                print(f"  Processing {model_name} | Fold {fold}...")

                # Load Model
                model = self._load_base_model(model_name, fold)

                # Extract Features (Mean, Std)
                f_val = self._predict_tta(model, val_loader)
                f_test = self._predict_tta(model, test_loader)

                val_feats_all.append(f_val)
                test_feats_all.append(f_test)

        # Concatenate all features: Shape (N, Num_Models * 2)
        X_val = np.concatenate(val_feats_all, axis=1)
        X_test = np.concatenate(test_feats_all, axis=1)

        # 3. Save to Cache
        print("Saving meta-features to cache...")
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        np.save(self.meta_features_val_path, X_val)
        np.save(self.meta_targets_val_path, y_val)
        np.save(self.meta_features_test_path, X_test)

        return X_val, y_val, X_test

    def train_meta_learner(self, X_val, y_val):
        """
        Trains a Logistic Regression meta-learner on the geometric features.
        """
        print("Training Meta-Learner (Logistic Regression)...")

        # Initialize and Train
        # Using liblinear for small binary classification tasks
        clf = LogisticRegression(solver="liblinear", random_state=Config.SEED, C=1.0)
        clf.fit(X_val, y_val)

        # Evaluate
        preds = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        print(f"Meta-Learner Validation AUC: {auc}")

        # Save Model
        joblib.dump(clf, Config.META_LEARNER_PATH)
        print(f"Meta-Learner saved to {Config.META_LEARNER_PATH}")

        return clf, preds

    def predict_stacking(self, X_test, test_ids):
        """
        Generates final predictions for the test set and saves the submission file.
        """
        print("Generating Final Predictions...")

        # Load Meta-Learner
        if not os.path.exists(Config.META_LEARNER_PATH):
            raise FileNotFoundError(
                f"Meta-learner not found at {Config.META_LEARNER_PATH}"
            )

        clf = joblib.load(Config.META_LEARNER_PATH)

        # Predict
        preds = clf.predict_proba(X_test)[:, 1]

        # Create Submission DataFrame
        df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

        # Save
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
