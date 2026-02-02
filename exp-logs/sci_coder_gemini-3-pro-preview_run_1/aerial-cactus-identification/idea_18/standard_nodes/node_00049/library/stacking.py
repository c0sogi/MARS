import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint, load_checkpoint
from library.dataset import load_data_to_memory, CactusDataset, get_transforms
from library.models import CactusModel
from library.engine import train_one_epoch, evaluate, SWAHandler

logger = get_logger("stacking")


class StackingTrainer:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.working_dir = Config.WORKING_DIR
        self.oof_cache_path = os.path.join(self.working_dir, "oof_predictions.parquet")
        self.test_preds_cache_path = os.path.join(
            self.working_dir, "test_predictions_raw.parquet"
        )
        self.val_preds_cache_path = os.path.join(
            self.working_dir, "val_predictions.parquet"
        )

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def load_all_data(self):
        """Loads all image data and metadata into memory."""
        logger.info("Loading Training Data...")
        self.train_imgs, self.train_labels, self.train_fsizes, self.train_ids = (
            load_data_to_memory(
                Config.TRAIN_METADATA,
                {
                    "imgs": "train_imgs",
                    "labels": "train_labels",
                    "fsizes": "train_fsizes",
                    "ids": "train_ids",
                },
            )
        )

        # Load Validation Data (merged into training for CV)
        # Note: In this pipeline, we merge train/val metadata to perform our own 5-fold CV
        # However, to respect the provided file structure, we load both and concatenate.
        logger.info("Loading Validation Data (merging for CV)...")
        val_imgs, val_labels, val_fsizes, val_ids = load_data_to_memory(
            Config.VAL_METADATA,
            {
                "imgs": None,
                "labels": None,
                "fsizes": None,
                "ids": None,
            },  # Don't cache separate val files to avoid conflict
            load_cached_data=False,
        )

        self.train_imgs = np.concatenate([self.train_imgs, val_imgs], axis=0)
        self.train_labels = np.concatenate([self.train_labels, val_labels], axis=0)
        self.train_fsizes = np.concatenate([self.train_fsizes, val_fsizes], axis=0)
        self.train_ids = np.concatenate([self.train_ids, val_ids], axis=0)

        logger.info(f"Total Training Samples for CV: {len(self.train_imgs)}")

        logger.info("Loading Test Data...")
        self.test_imgs, _, self.test_fsizes, self.test_ids = load_data_to_memory(
            Config.TEST_METADATA,
            {
                "imgs": "test_imgs",
                "fsizes": "test_fsizes",
                "ids": "test_ids",
            },
        )

        # Compute Global File Size Stats for FiLM Normalization
        self.fs_mean = np.mean(self.train_fsizes)
        self.fs_std = np.std(self.train_fsizes)
        self.fs_stats = (self.fs_mean, self.fs_std)
        logger.info(
            f"Global File Size Stats - Mean: {self.fs_mean:.4f}, Std: {self.fs_std:.4f}"
        )

    def get_base_model_predictions(self, load_cached_data=True):
        """
        Orchestrates the training of base models to generate OOF, Test, and Hold-out Val predictions.
        """
        if (
            load_cached_data
            and os.path.exists(self.oof_cache_path)
            and os.path.exists(self.test_preds_cache_path)
            and os.path.exists(self.val_preds_cache_path)
        ):
            logger.info(
                f"Loading cached OOF, Test, and Val predictions from {self.working_dir}..."
            )
            oof_df = pd.read_parquet(self.oof_cache_path)
            test_preds_df = pd.read_parquet(self.test_preds_cache_path)
            val_preds_df = pd.read_parquet(self.val_preds_cache_path)
            return oof_df, test_preds_df, val_preds_df

        logger.info("Starting Base Model Training (5-Fold CV)...")

        # Initialize containers
        # OOF DataFrame: initialized with IDs and Targets (Train Only)
        oof_df = pd.DataFrame({"id": self.train_ids, "target": self.train_labels})

        # Test Preds DataFrame
        test_preds_df = pd.DataFrame({"id": self.test_ids})

        # Val Preds DataFrame (Hold-out)
        val_preds_df = pd.DataFrame({"id": self.val_ids, "target": self.val_labels})

        # Setup Cross-Validation
        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Iterate over defined model configurations
        for model_name, config in Config.MODEL_CONFIGS.items():
            logger.info(f"=== Training Model Configuration: {model_name} ===")

            oof_preds = np.zeros(len(self.train_imgs))
            test_preds_accum = np.zeros(len(self.test_imgs))
            val_preds_accum = np.zeros(len(self.val_imgs))

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(self.train_imgs, self.train_labels)
            ):
                logger.info(f"--- Fold {fold+1}/{Config.NUM_FOLDS} ---")

                # Train single fold
                # Returns: OOF Probs, Test Probs, Hold-out Val Probs
                val_probs, test_probs, holdout_probs = self._train_single_fold(
                    model_name, config, fold, train_idx, val_idx
                )

                # Flatten predictions
                val_probs = val_probs.reshape(-1)
                test_probs = test_probs.reshape(-1)
                holdout_probs = holdout_probs.reshape(-1)

                # Store OOF predictions
                oof_preds[val_idx] = val_probs

                # Accumulate Test and Hold-out Val predictions
                test_preds_accum += test_probs
                val_preds_accum += holdout_probs

            # Average predictions across folds
            test_preds_avg = test_preds_accum / Config.NUM_FOLDS
            val_preds_avg = val_preds_accum / Config.NUM_FOLDS

            # Add to DataFrames
            oof_df[model_name] = oof_preds
            test_preds_df[model_name] = test_preds_avg
            val_preds_df[model_name] = val_preds_avg

            # Print Model Performance
            auc = roc_auc_score(self.train_labels, oof_preds)
            logger.info(f"Model {model_name} CV AUC: {auc:.6f}")

        # Save to Cache
        logger.info(f"Saving predictions to {self.working_dir}...")
        oof_df.to_parquet(self.oof_cache_path)
        test_preds_df.to_parquet(self.test_preds_cache_path)
        val_preds_df.to_parquet(self.val_preds_cache_path)

        return oof_df, test_preds_df, val_preds_df

    def _train_single_fold(self, model_name, config, fold, train_idx, val_idx):
        """Trains a single model on a single fold."""
        # Prepare Datasets
        train_dataset = CactusDataset(
            self.train_imgs[train_idx],
            self.train_labels[train_idx],
            self.train_fsizes[train_idx],
            self.train_ids[train_idx],
            transform=get_transforms("train", config["in_chans"]),
            in_chans=config["in_chans"],
            fs_stats=self.fs_stats,
        )
        val_dataset = CactusDataset(
            self.train_imgs[val_idx],
            self.train_labels[val_idx],
            self.train_fsizes[val_idx],
            self.train_ids[val_idx],
            transform=get_transforms("val", config["in_chans"]),
            in_chans=config["in_chans"],
            fs_stats=self.fs_stats,
        )
        test_dataset = CactusDataset(
            self.test_imgs,
            None,
            self.test_fsizes,
            self.test_ids,
            transform=get_transforms("test", config["in_chans"]),
            in_chans=config["in_chans"],
            fs_stats=self.fs_stats,
        )
        # Hold-out Validation Dataset (treated as Test)
        holdout_dataset = CactusDataset(
            self.val_imgs,
            self.val_labels,
            self.val_fsizes,
            self.val_ids,
            transform=get_transforms("test", config["in_chans"]),
            in_chans=config["in_chans"],
            fs_stats=self.fs_stats,
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = CactusModel(
            arch=config["arch"],
            in_chans=config["in_chans"],
            num_classes=Config.NUM_CLASSES,
            use_film=config["use_film"],
            use_mtl=config["use_mtl"],
        ).to(self.device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # SWA Handler
        swa_handler = SWAHandler(model, Config.SWA_START_EPOCH, self.device)

        # Training Loop
        best_auc = 0.0
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}_best.pth"
        )

        for epoch in range(Config.EPOCHS):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, self.device, epoch
            )

            # SWA Update
            if Config.USE_SWA:
                swa_handler.update(model, epoch)

            # Validation
            val_metrics, val_auc, _, _ = evaluate(model, val_loader, self.device)

            # Scheduler Step
            scheduler.step()

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_auc, best_model_path
                )

            # Logging (sparse to reduce clutter)
            if (epoch + 1) % 5 == 0 or (epoch + 1) == Config.EPOCHS:
                logger.info(
                    f"Epoch {epoch+1}/{Config.EPOCHS} | "
                    f"Train Loss: {train_metrics['Loss']['avg']:.4f} | "
                    f"Val AUC: {val_auc:.6f}"
                )

        # Load Best Model or SWA Model for Inference
        if Config.USE_SWA and swa_handler.active:
            logger.info("Using SWA Model for inference...")
            swa_handler.update_bn(train_loader)
            final_model = swa_handler.get_model()
        else:
            logger.info(f"Loading best model from {best_model_path}...")
            load_checkpoint(model, None, None, best_model_path, self.device)
            final_model = model

        # Switch to deploy mode (RepVGG fusion)
        final_model.eval()
        if hasattr(final_model, "module"):
            final_model.module.switch_to_deploy()
        else:
            final_model.switch_to_deploy()

        # Inference (TTA included in evaluate)
        _, val_auc_final, val_probs, _ = evaluate(final_model, val_loader, self.device)
        _, _, test_probs, _ = evaluate(final_model, test_loader, self.device)
        _, _, holdout_probs, _ = evaluate(final_model, holdout_loader, self.device)

        logger.info(f"Fold {fold+1} Final AUC: {val_auc_final:.6f}")

        return val_probs, test_probs, holdout_probs

    def train_meta_learner(self, oof_df):
        """
        Trains a Logistic Regression Meta-Learner on OOF predictions and metadata.
        """
        logger.info("Training Meta-Learner...")

        # Prepare Features
        # Features: [Prob_Model1, Prob_Model2, ..., FileSize_ZScore]
        model_cols = list(Config.MODEL_CONFIGS.keys())

        X = oof_df[model_cols].values
        y = oof_df["target"].values

        # Add File Size as feature (Standard Scaled)
        # We use the raw file sizes corresponding to the IDs
        # Since oof_df has 'id', we map back to self.train_fsizes using the order
        # But self.train_fsizes is aligned with self.train_ids.
        # oof_df was created aligned with self.train_ids.
        fsizes = self.train_fsizes.reshape(-1, 1)

        # Scale File Sizes for LR
        self.fs_scaler = StandardScaler()
        fsizes_scaled = self.fs_scaler.fit_transform(fsizes)

        X_full = np.hstack([X, fsizes_scaled])

        # Train Logistic Regression
        meta_model = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", random_state=Config.SEED
        )
        meta_model.fit(X_full, y)

        # Evaluate
        preds = meta_model.predict_proba(X_full)[:, 1]
        auc = roc_auc_score(y, preds)
        logger.info(f"Meta-Learner OOF AUC: {auc:.8f}")

        # Save Meta-Model
        joblib.dump(meta_model, os.path.join(self.working_dir, "meta_model.joblib"))

        return meta_model

    def generate_submission(self, meta_model, test_preds_df):
        """
        Generates final submission using the trained Meta-Learner.
        """
        logger.info("Generating Final Submission...")

        # Prepare Test Features
        model_cols = list(Config.MODEL_CONFIGS.keys())
        X_test = test_preds_df[model_cols].values

        # Add File Size Feature
        fsizes_test = self.test_fsizes.reshape(-1, 1)
        fsizes_test_scaled = self.fs_scaler.transform(fsizes_test)

        X_test_full = np.hstack([X_test, fsizes_test_scaled])

        # Predict
        final_probs = meta_model.predict_proba(X_test_full)[:, 1]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"id": test_preds_df["id"], "has_cactus": final_probs}
        )

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission.to_csv(save_path, index=False)
        logger.info(f"Submission saved to {save_path}")

        # Optional: Save stacked predictions for analysis
        stack_path = os.path.join(Config.SUBMISSION_DIR, "stacked_submission.csv")
        submission.to_csv(stack_path, index=False)

    def run(self):
        """Main execution flow."""
        seed_everything(Config.SEED)

        # 1. Load Data
        self.load_all_data()

        # 2. Base Models (Level 0)
        oof_df, test_preds_df, val_preds_df = self.get_base_model_predictions(
            load_cached_data=True
        )

        # 3. Meta Learner (Level 1)
        meta_model = self.train_meta_learner(oof_df)

        # 4. Submission
        self.generate_submission(meta_model, test_preds_df)
