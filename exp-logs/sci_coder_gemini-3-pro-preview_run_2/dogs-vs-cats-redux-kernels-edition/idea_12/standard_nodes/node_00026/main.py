import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from timm.data.mixup import Mixup

# Import library modules
from library.config import Config
from library.dataset import PetDataset, get_transforms
from library.models import PetModel
from library.engine import fit, evaluate, predict_tta
from library.utils import seed_everything, get_logger, generate_model_soup


def main():
    # 1. Setup
    seed_everything(Config.seed)
    Config.setup_directories()
    logger = get_logger("main")
    device = torch.device(Config.device)

    # Override Config for Fast Baseline
    Config.epochs = 3  # Reduced from 20
    Config.soup_epochs = 2  # Last 2 epochs
    TRAIN_SAMPLE_SIZE = 1500  # Subsample training data for speed

    logger.info(
        f"Starting Fast Baseline Run with {Config.epochs} epochs and {TRAIN_SAMPLE_SIZE} training samples."
    )

    # 2. Data Loading
    train_df_full = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    # Subsample training data for fast execution
    train_df = train_df_full.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(train_df_full)), random_state=Config.seed
    ).reset_index(drop=True)

    # Prepare storage for Stacking
    # OOF Predictions: (N_train_samples, N_models)
    # Val Predictions: (N_val_samples, N_models)
    # Test Predictions: (N_test_samples, N_models)
    n_models = len(Config.models)
    n_folds = Config.n_folds

    # We need to align OOF predictions with the original train_df indices
    oof_preds_dict = {
        model_name: np.zeros(len(train_df)) for model_name in Config.models
    }
    oof_targets_dict = {
        model_name: np.zeros(len(train_df)) for model_name in Config.models
    }

    # For Hold-out Validation and Test, we average predictions across folds for each model
    val_preds_dict = {
        model_name: np.zeros((len(val_df), n_folds)) for model_name in Config.models
    }
    test_preds_dict = {
        model_name: np.zeros((len(test_df), n_folds)) for model_name in Config.models
    }

    # Mixup function
    mixup_fn = Mixup(
        mixup_alpha=Config.mixup_alpha,
        cutmix_alpha=Config.cutmix_alpha,
        prob=Config.mixup_prob,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.0,
        num_classes=2,
    )

    # 3. Training Loop
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.seed)

    for model_idx, model_name in enumerate(Config.models):
        logger.info(f"Processing Model: {model_name}")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["label"])
        ):
            logger.info(f"  Fold {fold+1}/{n_folds}")

            # Split Data
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            # Datasets & Loaders
            train_ds = PetDataset(
                fold_train_df, mode="train", transforms=get_transforms("train")
            )
            val_ds = PetDataset(
                fold_val_df, mode="val", transforms=get_transforms("val")
            )
            # Hold-out Val Dataset (for this fold's model evaluation)
            holdout_ds = PetDataset(
                val_df, mode="val", transforms=get_transforms("val")
            )
            # Test Dataset
            test_ds = PetDataset(
                test_df, mode="test", transforms=get_transforms("test")
            )

            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=Config.batch_size,
                shuffle=True,
                num_workers=Config.num_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )
            holdout_loader = torch.utils.data.DataLoader(
                holdout_ds,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            # Model, Optimizer, Scheduler
            model = PetModel(model_name, pretrained=True).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.epochs, eta_min=Config.min_lr
            )

            # Train
            fit(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                Config.epochs,
                mixup_fn,
                Config,
                fold,
                model_name,
                patience=Config.epochs,
            )

            # Generate Model Soup
            # Identify checkpoints to average
            ckpt_paths = []
            for e in range(Config.epochs - Config.soup_epochs, Config.epochs):
                p = os.path.join(
                    Config.checkpoint_dir, f"{model_name}_fold_{fold}_epoch_{e}.pth"
                )
                if os.path.exists(p):
                    ckpt_paths.append(p)

            soup_path = os.path.join(
                Config.checkpoint_dir, f"soup_{model_name}_fold_{fold}.pth"
            )
            if ckpt_paths:
                soup_state = generate_model_soup(ckpt_paths, soup_path)
                model.load_state_dict(soup_state)
            else:
                logger.warning("No checkpoints found for soup. Using last state.")

            # Inference on OOF (Validation Fold)
            model.eval()
            val_res = evaluate(model, val_loader, device)
            oof_preds_dict[model_name][val_idx] = val_res["preds"].flatten()
            oof_targets_dict[model_name][val_idx] = val_res["targets"].flatten()

            # Inference on Hold-out Validation Set
            holdout_res = evaluate(model, holdout_loader, device)
            val_preds_dict[model_name][:, fold] = holdout_res["preds"].flatten()

            # Inference on Test Set (TTA)
            test_preds, _ = predict_tta(model, test_loader, device)
            test_preds_dict[model_name][:, fold] = test_preds

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                holdout_loader,
                test_loader,
            )
            torch.cuda.empty_cache()

    # 4. Stacking (Meta-Learner)
    logger.info("Training Meta-Learner...")

    # Prepare Training Data for Meta-Learner (OOF Predictions)
    # X_meta: (N_train, N_models)
    X_meta_train = np.column_stack([oof_preds_dict[m] for m in Config.models])
    y_meta_train = train_df["label"].values

    # Prepare Validation Data for Meta-Learner (Averaged Hold-out Predictions)
    # Average across folds for each model first
    val_preds_avg = {m: val_preds_dict[m].mean(axis=1) for m in Config.models}
    X_meta_val = np.column_stack([val_preds_avg[m] for m in Config.models])
    y_meta_val = val_df["label"].values

    # Prepare Test Data for Meta-Learner
    test_preds_avg = {m: test_preds_dict[m].mean(axis=1) for m in Config.models}
    X_meta_test = np.column_stack([test_preds_avg[m] for m in Config.models])

    # Train Logistic Regression
    meta_model = LogisticRegression(**Config.meta_learner_params)
    meta_model.fit(X_meta_train, y_meta_train)

    # 5. Final Evaluation
    final_val_probs = meta_model.predict_proba(X_meta_val)[:, 1]
    final_metric = log_loss(y_meta_val, final_val_probs)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate per-sample error
    epsilon = 1e-15
    preds_clipped = np.clip(final_val_probs, epsilon, 1 - epsilon)
    sample_losses = -(
        y_meta_val * np.log(preds_clipped)
        + (1 - y_meta_val) * np.log(1 - preds_clipped)
    )

    # Get image metadata for validation set
    widths, heights, ratios = [], [], []

    # Read a subset of validation images to save time if dataset is huge,
    # but requirement implies analysis on the set. 4500 is manageable.
    for _, row in val_df.iterrows():
        path = os.path.join(Config.input_dir, row["filepath"])
        img = cv2.imread(path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
        else:
            widths.append(np.nan)
            heights.append(np.nan)
            ratios.append(np.nan)

    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "width": widths,
            "height": heights,
            "aspect_ratio": ratios,
        }
    )

    # Drop NaNs if any read errors
    analysis_df = analysis_df.dropna()

    # Calculate correlations
    correlations = analysis_df.corr()["loss"].drop("loss")
    print("Correlation between Error (Log Loss) and Image Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.01366509944361823
    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} < {THRESHOLD}. Generating submission."
        )
        final_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]

        submission = pd.DataFrame({"id": test_df["id"], "label": final_test_probs})

        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(
            f"Validation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
