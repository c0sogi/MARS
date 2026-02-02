import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import create_folds, get_tokenizer, QuestDataset
from library.models import SegmentAwareNet
from library.engine import train_model, extract_features
from library.stacking import StackingTrainer


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We reduce epochs to ensure completion within the time limit.
    Config.EPOCHS = 2

    print("=" * 60)
    print(" STARTING PIPELINE")
    print(f" Device: {Config.DEVICE}")
    print(f" Epochs: {Config.EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[Data] Loading Metadata...")

    # Load Training Data with Folds
    train_df = create_folds(load_cached_data=True)

    # Load Hold-out Validation Data
    val_meta_path = "./metadata/val_metadata.csv"
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")
    val_holdout_df = pd.read_csv(val_meta_path)

    # Load Test Data
    test_meta_path = "./metadata/test_metadata.csv"
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")
    test_df = pd.read_csv(test_meta_path)

    print(f"Train Shape: {train_df.shape}")
    print(f"Val Holdout Shape: {val_holdout_df.shape}")
    print(f"Test Shape: {test_df.shape}")

    # Prepare Targets
    train_targets = train_df[Config.TARGET_COLS].values
    val_holdout_targets = val_holdout_df[Config.TARGET_COLS].values

    # --------------------------------------------------------------------------
    # 3. Level 1: Backbone Training & Feature Extraction
    # --------------------------------------------------------------------------
    # Storage for L1 outputs
    # Structure: {backbone_name: {'oof_preds': ..., 'val_preds': ..., 'test_preds': ...}}
    l1_outputs = {}

    # Initialize Stacking Trainer
    stacker = StackingTrainer()

    for backbone_key, backbone_cfg in Config.BACKBONES.items():
        print(f"\n>>> Processing Backbone: {backbone_key} ({backbone_cfg['name']})")

        tokenizer = get_tokenizer(backbone_cfg["name"])

        # Get Gradient Accumulation Steps (Default to 1 if not set)
        grad_accum_steps = backbone_cfg.get("grad_accum_steps", 1)
        print(f"  Batch Size: {backbone_cfg['batch_size']}")
        print(f"  Grad Accum Steps: {grad_accum_steps}")

        # Initialize Feature Arrays
        # We don't know feature dim yet, will init on first fold
        oof_features = None
        val_holdout_features_accum = None
        test_features_accum = None

        # Cross-Validation Loop
        for fold in range(Config.N_FOLDS):
            print(f"  [Fold {fold}/{Config.N_FOLDS - 1}]")

            # Split Data
            train_idx = train_df[train_df["fold"] != fold].index
            val_idx = train_df[train_df["fold"] == fold].index

            df_train_fold = train_df.loc[train_idx].reset_index(drop=True)
            df_val_fold = train_df.loc[val_idx].reset_index(drop=True)

            # Create Datasets
            train_ds = QuestDataset(df_train_fold, tokenizer, max_len=Config.MAX_LEN)
            val_ds = QuestDataset(df_val_fold, tokenizer, max_len=Config.MAX_LEN)

            # Create DataLoaders
            train_loader = DataLoader(
                train_ds,
                batch_size=backbone_cfg["batch_size"],
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=backbone_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            model = SegmentAwareNet(backbone_cfg["name"], pretrained=True)
            model.to(Config.DEVICE)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=backbone_cfg["lr"],
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Adjust T_max for Gradient Accumulation
            num_update_steps_per_epoch = len(train_loader) // grad_accum_steps
            if len(train_loader) % grad_accum_steps != 0:
                num_update_steps_per_epoch += 1

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS * num_update_steps_per_epoch
            )

            # Train
            save_filename = f"{backbone_key}_fold{fold}.pth"
            train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                Config.DEVICE,
                scheduler,
                Config.EPOCHS,
                save_filename,
                grad_accum_steps=grad_accum_steps,
            )

            # Load Best Model for Inference
            checkpoint = torch.load(
                os.path.join(Config.WORKING_DIR, save_filename),
                map_location=Config.DEVICE,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # --- Feature Extraction ---

            # 1. OOF Features (Validation Fold)
            # We use the same val_loader
            feats_val, _ = extract_features(val_loader, model, Config.DEVICE)

            # Initialize OOF array if needed
            if oof_features is None:
                feature_dim = feats_val.shape[1]
                oof_features = np.zeros((len(train_df), feature_dim), dtype=np.float32)

            # Fill OOF
            oof_features[val_idx] = feats_val

            # 2. Holdout Validation Features
            val_holdout_ds = QuestDataset(
                val_holdout_df, tokenizer, max_len=Config.MAX_LEN, inference=True
            )
            val_holdout_loader = DataLoader(
                val_holdout_ds,
                batch_size=backbone_cfg["batch_size"] * 2,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            feats_holdout, _ = extract_features(
                val_holdout_loader, model, Config.DEVICE
            )

            if val_holdout_features_accum is None:
                val_holdout_features_accum = np.zeros_like(feats_holdout)
            val_holdout_features_accum += feats_holdout

            # 3. Test Features
            test_ds = QuestDataset(
                test_df, tokenizer, max_len=Config.MAX_LEN, inference=True
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=backbone_cfg["batch_size"] * 2,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            feats_test, _ = extract_features(test_loader, model, Config.DEVICE)

            if test_features_accum is None:
                test_features_accum = np.zeros_like(feats_test)
            test_features_accum += feats_test

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                val_holdout_loader,
                test_loader,
            )
            torch.cuda.empty_cache()
            gc.collect()

        # Average Features across folds
        val_holdout_features_avg = val_holdout_features_accum / Config.N_FOLDS
        test_features_avg = test_features_accum / Config.N_FOLDS

        # Train L1 Ridge and Predict
        # We need predictions for both Holdout Val and Test.
        # The StackingTrainer.train_l1_ridge method trains on (features, targets) and predicts on 'test_features'.

        # 1. Predict on Holdout Val
        print(f"  Training L1 Ridge for {backbone_key} (Target: Holdout Val)...")
        res_val = stacker.train_l1_ridge(
            backbone_key,
            oof_features,
            train_targets,
            val_holdout_features_avg,
            train_df["fold"].values,
            load_cached=False,
        )

        # 2. Predict on Test
        # Note: This effectively retrains the Ridge model. Since Ridge is fast, this is acceptable
        # to avoid modifying the library code.
        print(f"  Training L1 Ridge for {backbone_key} (Target: Test)...")
        res_test = stacker.train_l1_ridge(
            backbone_key,
            oof_features,
            train_targets,
            test_features_avg,
            train_df["fold"].values,
            load_cached=False,
        )

        # Store results
        l1_outputs[backbone_key] = {
            "oof_preds": res_val["oof_preds"],
            "val_preds": res_val[
                "test_preds"
            ],  # This contains predictions for val_holdout_features_avg
            "test_preds": res_test[
                "test_preds"
            ],  # This contains predictions for test_features_avg
        }

        # Free memory
        del (
            oof_features,
            val_holdout_features_accum,
            test_features_accum,
            val_holdout_features_avg,
            test_features_avg,
        )
        gc.collect()

    # --------------------------------------------------------------------------
    # 4. Level 2: Meta-Learner Stacking
    # --------------------------------------------------------------------------
    print("\n>>> Level 2: Meta-Learner Stacking")

    # Prepare inputs for Meta-Learner (Holdout Evaluation)
    l1_outputs_for_val = {
        k: {"oof_preds": v["oof_preds"], "test_preds": v["val_preds"]}
        for k, v in l1_outputs.items()
    }

    # Train Meta-Learner and predict on Holdout Val
    print("  Generating Holdout Validation Predictions...")
    val_final_preds = stacker.train_l2_meta(
        l1_outputs_for_val, train_targets, load_cached=False
    )

    # Prepare inputs for Meta-Learner (Test Submission)
    l1_outputs_for_test = {
        k: {"oof_preds": v["oof_preds"], "test_preds": v["test_preds"]}
        for k, v in l1_outputs.items()
    }

    # Train Meta-Learner and predict on Test
    print("  Generating Test Predictions...")
    test_final_preds = stacker.train_l2_meta(
        l1_outputs_for_test, train_targets, load_cached=False
    )

    # --------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" VALIDATION & ANALYSIS")
    print("=" * 60)

    # Compute Metric
    val_score = compute_spearmanr(val_holdout_targets, val_final_preds)
    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(val_holdout_targets - val_final_preds), axis=1)

    # Add error to dataframe for correlation
    val_holdout_df["error_magnitude"] = mae_per_sample

    # Compute text lengths
    val_holdout_df["question_len"] = (
        val_holdout_df["question_body"].fillna("").str.len()
    )
    val_holdout_df["answer_len"] = val_holdout_df["answer"].fillna("").str.len()

    # Correlations
    corr_q = val_holdout_df["error_magnitude"].corr(val_holdout_df["question_len"])
    corr_a = val_holdout_df["error_magnitude"].corr(val_holdout_df["answer_len"])

    print(f"Correlation between Error and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length:   {corr_a:.4f}")

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.40698660691461275

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        stacker.save_submission(test_final_preds, test_df["qa_id"].values)
    else:
        print(
            f"\nValidation score ({val_score}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
