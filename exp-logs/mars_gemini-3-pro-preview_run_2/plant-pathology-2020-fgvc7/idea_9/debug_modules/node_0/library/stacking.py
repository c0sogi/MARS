import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib

from library.config import Config
from library.model import AppleDiseaseModel
from library.dataset import get_dataloaders, prepare_data, AppleDataset, get_transforms
from library.utils import seed_everything


class StackingPipeline:
    """
    Implements Stacked Generalization for Apple Disease Detection.
    Generates OOF features, trains a meta-learner, and produces final predictions.
    """

    def __init__(self):
        self.device = Config.device
        self.models_config = Config.models
        self.num_folds = Config.num_folds
        self.working_dir = Config.working_dir
        self.meta_model_path = os.path.join(self.working_dir, "meta_learner_lr.joblib")

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _load_model(self, model_cfg, fold):
        """
        Loads a model checkpoint for a specific configuration and fold.
        Prioritizes SWA models if configured and available.
        """
        model_name = model_cfg["name"]

        # Determine checkpoint path
        if Config.use_swa:
            ckpt_path = os.path.join(
                self.working_dir, f"swa_model_{model_name}_fold_{fold}.pth"
            )
            if not os.path.exists(ckpt_path):
                # Fallback to best model if SWA not found
                ckpt_path = os.path.join(
                    self.working_dir, f"best_model_{model_name}_fold_{fold}.pth"
                )
        else:
            ckpt_path = os.path.join(
                self.working_dir, f"best_model_{model_name}_fold_{fold}.pth"
            )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

        # Initialize model
        model = AppleDiseaseModel(
            model_name=model_name,
            pretrained=False,  # No need to download weights, we load state_dict
            num_classes=Config.num_classes,
            drop_rate=model_cfg["dropout_rate"],
            drop_path_rate=model_cfg["drop_path_rate"],
            use_gem=model_cfg["use_gem"],
        )

        # Load weights
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def _predict(self, model, dataloader):
        """
        Runs inference on a dataloader and returns probabilities.
        """
        preds = []
        image_ids = []

        with torch.no_grad():
            for images, _, ids in dataloader:
                images = images.to(self.device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())
                image_ids.extend(ids)

        return np.concatenate(preds, axis=0), image_ids

    def get_oof_predictions(self, load_cached_data=True):
        """
        Generates or loads Out-Of-Fold predictions for the training set.
        These serve as features for the meta-learner.
        """
        cache_path = os.path.join(self.working_dir, "oof_predictions.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached OOF predictions from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Generating OOF predictions...")
        seed_everything(Config.seed)

        # We need to aggregate predictions for all samples
        # Since we use StratifiedKFold in get_dataloaders, we iterate folds to cover all data

        # Dictionary to store results: image_id -> {feature_col: val, target_col: val}
        results = {}

        for fold in range(self.num_folds):
            print(f"  Processing Fold {fold}/{self.num_folds - 1}")

            # Get validation loader for this fold (using first model config for transforms/batch size is fine for targets/ids)
            # But we need specific loaders for specific image sizes of different models

            # 1. Get Targets and IDs from the first model's loader (ground truth is same)
            _, val_loader_ref, _ = get_dataloaders(
                fold, self.models_config[0]["image_size"], Config.batch_size
            )

            # Store Ground Truth
            # We iterate the loader to extract targets aligned with image_ids
            for _, targets, ids in val_loader_ref:
                targets = targets.numpy()
                for i, img_id in enumerate(ids):
                    if img_id not in results:
                        results[img_id] = {
                            "image_id": img_id,
                            "target_rust": targets[i, 0],
                            "target_scab": targets[i, 1],
                        }

            # 2. Generate Predictions for each model architecture
            for m_idx, model_cfg in enumerate(self.models_config):
                # Get loader with correct image size
                _, val_loader, _ = get_dataloaders(
                    fold, model_cfg["image_size"], Config.batch_size
                )

                # Load Model
                model = self._load_model(model_cfg, fold)

                # Predict
                probs, ids = self._predict(model, val_loader)

                # Store Predictions
                col_prefix = f"model_{m_idx}"
                for i, img_id in enumerate(ids):
                    results[img_id][f"{col_prefix}_rust"] = probs[i, 0]
                    results[img_id][f"{col_prefix}_scab"] = probs[i, 1]

        # Convert to DataFrame
        oof_df = pd.DataFrame.from_dict(results, orient="index").reset_index(drop=True)

        # Save to cache
        oof_df.to_parquet(cache_path, index=False)
        print(f"OOF predictions saved to {cache_path}")

        return oof_df

    def get_test_predictions(self, load_cached_data=True):
        """
        Generates or loads Test set predictions.
        Averages predictions across folds for each model architecture (Bagging).
        """
        cache_path = os.path.join(self.working_dir, "test_base_predictions.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Test predictions from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Generating Test predictions...")
        seed_everything(Config.seed)

        # Load Test Data (just to get IDs)
        _, test_df = prepare_data()
        image_ids = test_df["image_id"].values

        # Initialize storage for averaged predictions: shape (n_samples, n_models, n_classes)
        # We have 2 classes (Rust, Scab)
        num_samples = len(image_ids)
        num_models = len(self.models_config)
        avg_preds = np.zeros((num_samples, num_models, 2), dtype=np.float32)

        for m_idx, model_cfg in enumerate(self.models_config):
            print(f"  Processing Model {m_idx}: {model_cfg['name']}")

            # Accumulator for this model across folds
            model_fold_preds = np.zeros((num_samples, 2), dtype=np.float32)

            # Get Test Loader (same for all folds, just depends on image size)
            test_dataset = AppleDataset(
                test_df,
                transforms=get_transforms("test", model_cfg["image_size"]),
                debug=Config.debug,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            for fold in range(self.num_folds):
                model = self._load_model(model_cfg, fold)
                probs, _ = self._predict(model, test_loader)
                model_fold_preds += probs

            # Average across folds
            avg_preds[:, m_idx, :] = model_fold_preds / self.num_folds

        # Construct DataFrame
        test_res = {"image_id": image_ids}
        for m_idx in range(num_models):
            col_prefix = f"model_{m_idx}"
            test_res[f"{col_prefix}_rust"] = avg_preds[:, m_idx, 0]
            test_res[f"{col_prefix}_scab"] = avg_preds[:, m_idx, 1]

        test_pred_df = pd.DataFrame(test_res)

        # Save to cache
        test_pred_df.to_parquet(cache_path, index=False)
        print(f"Test predictions saved to {cache_path}")

        return test_pred_df

    def train_meta_learner(self):
        """
        Trains the Logistic Regression meta-learner on OOF predictions.
        """
        print("Training Meta-Learner...")
        oof_df = self.get_oof_predictions(load_cached_data=True)

        # Define Features and Targets
        # Features: All model outputs [model_0_rust, model_0_scab, model_1_rust, ...]
        feature_cols = [c for c in oof_df.columns if c.startswith("model_")]
        target_cols = ["target_rust", "target_scab"]

        X = oof_df[feature_cols].values
        y = oof_df[target_cols].values

        # Initialize Meta-Learner
        # We use MultiOutputClassifier to handle the 2 binary targets with one estimator per target
        base_lr = LogisticRegression(**Config.meta_learner_params)
        meta_model = MultiOutputClassifier(base_lr, n_jobs=-1)

        # Train
        meta_model.fit(X, y)

        # Evaluate on Training Data (OOF) - Just for sanity check
        # Note: This is technically training score, but since input is OOF, it approximates CV score
        preds = meta_model.predict_proba(X)
        # predict_proba returns a list of arrays (one per target)
        # preds[0] is (n_samples, 2) for target 0. We want the probability of class 1.
        y_pred_rust = preds[0][:, 1]
        y_pred_scab = preds[1][:, 1]

        y_pred_combined = np.stack([y_pred_rust, y_pred_scab], axis=1)
        score = 0
        try:
            from sklearn.metrics import roc_auc_score

            score = roc_auc_score(y, y_pred_combined, average="macro")
            print(f"Meta-Learner OOF ROC AUC: {score:.6f}")
        except Exception as e:
            print(f"Could not calculate metric: {e}")

        # Save Meta-Learner
        joblib.dump(meta_model, self.meta_model_path)
        print(f"Meta-learner saved to {self.meta_model_path}")

        return meta_model

    def generate_submission(self):
        """
        Generates the final submission file using the trained meta-learner.
        """
        print("Generating Submission...")

        # 1. Get Test Features
        test_df = self.get_test_predictions(load_cached_data=True)
        feature_cols = [c for c in test_df.columns if c.startswith("model_")]
        X_test = test_df[feature_cols].values
        image_ids = test_df["image_id"].values

        # 2. Load Meta-Learner
        if not os.path.exists(self.meta_model_path):
            self.train_meta_learner()

        meta_model = joblib.load(self.meta_model_path)

        # 3. Predict Probabilities (Has Rust, Has Scab)
        preds_proba = meta_model.predict_proba(X_test)
        # preds_proba is list of [n_samples, 2] arrays. Extract prob of class 1.
        p_rust = preds_proba[0][:, 1]
        p_scab = preds_proba[1][:, 1]

        # 4. Reconstruct 4-Class Probabilities
        # Logic:
        # Healthy: No Rust AND No Scab
        # Multiple: Rust AND Scab
        # Rust: Rust AND No Scab
        # Scab: No Rust AND Scab
        # We assume independence for reconstruction: P(A and B) = P(A) * P(B)

        p_healthy = (1 - p_rust) * (1 - p_scab)
        p_multiple = p_rust * p_scab
        p_rust_only = p_rust * (1 - p_scab)
        p_scab_only = (1 - p_rust) * p_scab

        # Stack
        final_preds = np.stack(
            [p_healthy, p_multiple, p_rust_only, p_scab_only], axis=1
        )

        # Normalize to ensure sum = 1
        row_sums = final_preds.sum(axis=1, keepdims=True)
        final_preds = final_preds / row_sums

        # 5. Create Submission DataFrame
        # Columns based on sample_submission.csv content provided in description:
        # image_id, healthy, multiple_diseases, rust, scab
        submission_df = pd.DataFrame(
            {
                "image_id": image_ids,
                "healthy": final_preds[:, 0],
                "multiple_diseases": final_preds[:, 1],
                "rust": final_preds[:, 2],
                "scab": final_preds[:, 3],
            }
        )

        # Save
        submission_path = Config.submission_path
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # Print first few rows for verification
        print(submission_df.head())
