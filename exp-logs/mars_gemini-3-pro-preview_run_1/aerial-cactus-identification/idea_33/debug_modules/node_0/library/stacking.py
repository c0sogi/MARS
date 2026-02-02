import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.engine import Engine
from library.models import CactusRepVGG, CactusResNet


class GeometricFeatureExtractor:
    """
    Extracts geometric consistency features from TTA predictions.
    """

    @staticmethod
    def extract(tta_probs):
        """
        Computes Mean and Standard Deviation across TTA views.

        Args:
            tta_probs (np.ndarray): Shape (N, 4) containing probabilities for
                                    Original, HFlip, VFlip, Rotate180.

        Returns:
            np.ndarray: Shape (N, 2) containing [Mean, Std].
        """
        # Calculate Mean and Std along the view axis (axis 1)
        mean = np.mean(tta_probs, axis=1, keepdims=True)
        std = np.std(tta_probs, axis=1, keepdims=True)

        # Concatenate to form (N, 2)
        return np.hstack([mean, std])


class StackingEnsemble:
    """
    Implements Heterogeneous Geometric-Consistency Stacking.
    Manages model loading, feature extraction, caching, and meta-learner training.
    """

    def __init__(self):
        # Logistic Regression as the Meta-Learner
        # Solver 'liblinear' is good for small datasets
        self.meta_learner = LogisticRegression(
            solver="liblinear", random_state=Config.SEED, C=1.0
        )

        # Mapping for architecture instantiation
        self.model_archs = {"CactusRepVGG": CactusRepVGG, "CactusResNet": CactusResNet}

    def _load_model(self, model_path, device):
        """
        Loads a specific model checkpoint.
        """
        # Determine architecture from filename
        arch_cls = None
        for name, cls in self.model_archs.items():
            if name in model_path:
                arch_cls = cls
                break

        if arch_cls is None:
            raise ValueError(
                f"Could not determine architecture for model: {model_path}"
            )

        # Instantiate model
        model = arch_cls(num_classes=Config.NUM_CLASSES)

        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Structural Re-parameterization for RepVGG
        if "RepVGG" in model_path:
            model.switch_to_deploy()

        return model

    def get_meta_features(
        self, model_paths, loader, device, phase, load_cached_data=True
    ):
        """
        Generates or loads meta-features (Mean, Std) for a list of models.

        Args:
            model_paths (list): List of paths to .pth checkpoints.
            loader (DataLoader): DataLoader for the dataset.
            device (torch.device): Compute device.
            phase (str): 'val' or 'test' (used for cache naming).
            load_cached_data (bool): Whether to use cached features.

        Returns:
            np.ndarray: Matrix of shape (N, 2 * num_models).
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"meta_features_{phase}.npy")

        # 1. Attempt to load from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                print(f"Loading cached meta-features for {phase} from {cache_path}...")
                try:
                    return np.load(cache_path)
                except Exception as e:
                    print(f"Failed to load cache: {e}. Regenerating...")

        # 2. Compute from scratch
        print(f"Generating meta-features for {phase} (Models: {len(model_paths)})...")

        all_model_features = []

        for path in model_paths:
            print(f"  Processing model: {os.path.basename(path)}")

            # Load model
            model = self._load_model(path, device)

            # Get Raw TTA Probabilities (N, 4)
            # predict_tta_raw handles the 4 geometric views
            raw_probs, _ = Engine.predict_tta_raw(model, loader, device)

            # Extract Geometric Features (N, 2)
            features = GeometricFeatureExtractor.extract(raw_probs)
            all_model_features.append(features)

            # Cleanup to save VRAM
            del model
            torch.cuda.empty_cache()

        # Concatenate all features: (N, 2 * M)
        meta_features = np.hstack(all_model_features)

        # 3. Save to cache
        print(f"Saving meta-features to {cache_path}...")
        np.save(cache_path, meta_features)

        return meta_features

    def fit(self, model_paths, val_loader, val_targets, device):
        """
        Trains the Meta-Learner on validation set features.

        Args:
            model_paths (list): List of base model checkpoints.
            val_loader (DataLoader): Validation data loader.
            val_targets (np.ndarray): Ground truth labels for validation set.
            device (torch.device): Compute device.

        Returns:
            float: Validation AUC score.
        """
        print("Preparing Validation Meta-Features...")
        X_val = self.get_meta_features(model_paths, val_loader, device, phase="val")
        y_val = val_targets

        print(f"Training Meta-Learner on shape {X_val.shape}...")
        self.meta_learner.fit(X_val, y_val)

        # Evaluate
        preds = self.meta_learner.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        print(f"Meta-Learner Validation AUC: {auc:.10f}")
        return auc

    def predict(self, model_paths, test_loader, device):
        """
        Generates predictions for the test set using the trained Meta-Learner.

        Args:
            model_paths (list): List of base model checkpoints.
            test_loader (DataLoader): Test data loader.
            device (torch.device): Compute device.

        Returns:
            np.ndarray: Final probabilities (N,).
        """
        print("Preparing Test Meta-Features...")
        X_test = self.get_meta_features(model_paths, test_loader, device, phase="test")

        print(f"Predicting on Test set with shape {X_test.shape}...")
        preds = self.meta_learner.predict_proba(X_test)[:, 1]
        return preds

    def create_submission(
        self,
        model_paths,
        test_loader,
        test_ids,
        device,
        output_path=Config.SUBMISSION_PATH,
    ):
        """
        Generates predictions and saves the submission CSV.

        Args:
            model_paths (list): List of base model checkpoints.
            test_loader (DataLoader): Test data loader.
            test_ids (np.ndarray): Array of test image IDs.
            device (torch.device): Compute device.
            output_path (str): Path to save the CSV.
        """
        preds = self.predict(model_paths, test_loader, device)

        # Create DataFrame
        df = pd.DataFrame({"id": test_ids, "has_cactus": preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    def save_meta_learner(self, path):
        """Saves the trained meta-learner."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.meta_learner, path)

    def load_meta_learner(self, path):
        """Loads a trained meta-learner."""
        self.meta_learner = joblib.load(path)
