import sys
import os
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, logging

# Import library modules
from library.configuration import Config
from library.utilities import set_seed, get_optimizer_params
from library.features import SVDFeatureExtractor
from library.data import create_loaders
from library.architecture import HybridModel
from library.trainer import Trainer

# Suppress warnings and Transformers logging
logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override epochs to ensure fast baseline execution within time limits
    # 3 epochs is sufficient for convergence on this small dataset (~3k samples)
    Config.epochs = 3

    set_seed(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Feature Extraction
    # -------------------------------------------------------------------------
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Generate/Load SVD Features
    print("Processing Structural Features...")
    feature_extractor = SVDFeatureExtractor()
    # train_svd: (N_train, 256), val_svd: (N_val, 256), test_svd: (N_test, 256)
    train_svd, val_svd, test_svd = feature_extractor.process(load_cached_data=True)

    # Prepare Tokenizers
    print("Initializing Tokenizers...")
    tokenizer_a = AutoTokenizer.from_pretrained(Config.model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(Config.model_b_name)

    # -------------------------------------------------------------------------
    # 3. Stage 1: Teacher Training (Independent)
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: Teacher Training ===")

    # Accumulators for Test Logits (for distillation)
    teacher_a_test_logits = np.zeros((len(test_df), 1))
    teacher_b_test_logits = np.zeros((len(test_df), 1))

    # Stratified K-Fold for Training
    skf = StratifiedKFold(
        n_splits=Config.num_folds, shuffle=True, random_state=Config.seed
    )

    for fold, (train_idx, _) in enumerate(skf.split(train_df, train_df["Insult"])):
        print(f"\n--- Fold {fold + 1}/{Config.num_folds} ---")

        # Subset Data for this fold
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_train_svd = train_svd[train_idx]

        # --- Train Teacher A (DeBERTa) ---
        print(f"Training Teacher A ({Config.model_a_name})...")

        train_loader_a = create_loaders(
            fold_train_df,
            fold_train_svd,
            tokenizer_a,
            Config.train_batch_size,
            shuffle=True,
        )
        val_loader_a = create_loaders(
            val_df, val_svd, tokenizer_a, Config.valid_batch_size, shuffle=False
        )
        test_loader_a = create_loaders(
            test_df, test_svd, tokenizer_a, Config.valid_batch_size, shuffle=False
        )

        model_a = HybridModel(Config.model_a_name).to(device)
        optimizer_params = get_optimizer_params(
            model_a, Config.lr_backbone, Config.lr_head, Config.weight_decay
        )
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.epochs * len(train_loader_a)
        )

        trainer_a = Trainer(
            Config, model_a, train_loader_a, val_loader_a, optimizer, scheduler, device
        )
        trainer_a.fit(epochs=Config.epochs)

        # Predict Test Logits
        logits_a = trainer_a.predict(test_loader_a, return_logits=True)
        teacher_a_test_logits += logits_a / Config.num_folds

        # Cleanup
        del (
            model_a,
            trainer_a,
            optimizer,
            scheduler,
            train_loader_a,
            val_loader_a,
            test_loader_a,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # --- Train Teacher B (RoBERTa) ---
        print(f"Training Teacher B ({Config.model_b_name})...")

        train_loader_b = create_loaders(
            fold_train_df,
            fold_train_svd,
            tokenizer_b,
            Config.train_batch_size,
            shuffle=True,
        )
        val_loader_b = create_loaders(
            val_df, val_svd, tokenizer_b, Config.valid_batch_size, shuffle=False
        )
        test_loader_b = create_loaders(
            test_df, test_svd, tokenizer_b, Config.valid_batch_size, shuffle=False
        )

        model_b = HybridModel(Config.model_b_name).to(device)
        optimizer_params = get_optimizer_params(
            model_b, Config.lr_backbone, Config.lr_head, Config.weight_decay
        )
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.epochs * len(train_loader_b)
        )

        trainer_b = Trainer(
            Config, model_b, train_loader_b, val_loader_b, optimizer, scheduler, device
        )
        trainer_b.fit(epochs=Config.epochs)

        # Predict Test Logits
        logits_b = trainer_b.predict(test_loader_b, return_logits=True)
        teacher_b_test_logits += logits_b / Config.num_folds

        # Cleanup
        del (
            model_b,
            trainer_b,
            optimizer,
            scheduler,
            train_loader_b,
            val_loader_b,
            test_loader_b,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. Stage 2: Soft Label Generation
    # -------------------------------------------------------------------------
    print("\n=== Stage 2: Soft Label Generation ===")

    # Apply Temperature Scaling and Sigmoid
    T = Config.distillation_temp

    # Soft Targets for Student B (from Teacher A)
    soft_targets_for_b = 1.0 / (1.0 + np.exp(-teacher_a_test_logits / T))

    # Soft Targets for Student A (from Teacher B)
    soft_targets_for_a = 1.0 / (1.0 + np.exp(-teacher_b_test_logits / T))

    # Prepare Combined Data for Students
    # Structure: [Train Data (Hard Labels)] + [Test Data (Soft Labels)]

    # Train Data Labels
    train_labels = train_df["Insult"].values.astype(float).reshape(-1, 1)

    # Combined Labels
    labels_student_a = np.vstack([train_labels, soft_targets_for_a]).flatten()
    labels_student_b = np.vstack([train_labels, soft_targets_for_b]).flatten()

    # Combined DataFrames and SVD
    combined_df = pd.concat(
        [train_df[["Comment"]], test_df[["Comment"]]], axis=0
    ).reset_index(drop=True)
    combined_svd = np.vstack([train_svd, test_svd])

    print(f"Combined Dataset Size: {len(combined_df)}")

    # -------------------------------------------------------------------------
    # 5. Stage 3: Student Training (Mutual Distillation)
    # -------------------------------------------------------------------------
    print("\n=== Stage 3: Student Training ===")

    # Accumulators for Final Predictions
    student_a_val_preds = np.zeros((len(val_df), 1))
    student_b_val_preds = np.zeros((len(val_df), 1))

    student_a_test_preds = np.zeros((len(test_df), 1))
    student_b_test_preds = np.zeros((len(test_df), 1))

    for fold, (train_idx, _) in enumerate(skf.split(train_df, train_df["Insult"])):
        print(f"\n--- Student Fold {fold + 1}/{Config.num_folds} ---")

        # Construct Training Indices for Combined Dataset
        # Indices 0 to len(train_df)-1 correspond to Train Data
        # Indices len(train_df) to end correspond to Test Data

        # Fold Train Indices (from original train split)
        fold_train_indices = train_idx

        # All Test Indices (offset by len(train_df))
        test_indices = np.arange(len(train_df), len(combined_df))

        # Combine
        combined_train_idx = np.concatenate([fold_train_indices, test_indices])

        # Subset Combined Data
        fold_combined_df = combined_df.iloc[combined_train_idx].reset_index(drop=True)
        fold_combined_svd = combined_svd[combined_train_idx]

        # Subset Labels
        fold_labels_a = labels_student_a[combined_train_idx]
        fold_labels_b = labels_student_b[combined_train_idx]

        # --- Train Student A (DeBERTa) ---
        print(f"Training Student A ({Config.model_a_name})...")

        train_loader_a = create_loaders(
            fold_combined_df,
            fold_combined_svd,
            tokenizer_a,
            Config.train_batch_size,
            labels=fold_labels_a,
            shuffle=True,
        )
        val_loader_a = create_loaders(
            val_df, val_svd, tokenizer_a, Config.valid_batch_size, shuffle=False
        )
        test_loader_a = create_loaders(
            test_df, test_svd, tokenizer_a, Config.valid_batch_size, shuffle=False
        )

        model_a = HybridModel(Config.model_a_name).to(device)
        optimizer_params = get_optimizer_params(
            model_a, Config.lr_backbone, Config.lr_head, Config.weight_decay
        )
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.epochs * len(train_loader_a)
        )

        trainer_a = Trainer(
            Config, model_a, train_loader_a, val_loader_a, optimizer, scheduler, device
        )
        trainer_a.fit(epochs=Config.epochs)

        # Predict Val & Test
        student_a_val_preds += trainer_a.predict(val_loader_a) / Config.num_folds
        student_a_test_preds += trainer_a.predict(test_loader_a) / Config.num_folds

        del (
            model_a,
            trainer_a,
            optimizer,
            scheduler,
            train_loader_a,
            val_loader_a,
            test_loader_a,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # --- Train Student B (RoBERTa) ---
        print(f"Training Student B ({Config.model_b_name})...")

        train_loader_b = create_loaders(
            fold_combined_df,
            fold_combined_svd,
            tokenizer_b,
            Config.train_batch_size,
            labels=fold_labels_b,
            shuffle=True,
        )
        val_loader_b = create_loaders(
            val_df, val_svd, tokenizer_b, Config.valid_batch_size, shuffle=False
        )
        test_loader_b = create_loaders(
            test_df, test_svd, tokenizer_b, Config.valid_batch_size, shuffle=False
        )

        model_b = HybridModel(Config.model_b_name).to(device)
        optimizer_params = get_optimizer_params(
            model_b, Config.lr_backbone, Config.lr_head, Config.weight_decay
        )
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.epochs * len(train_loader_b)
        )

        trainer_b = Trainer(
            Config, model_b, train_loader_b, val_loader_b, optimizer, scheduler, device
        )
        trainer_b.fit(epochs=Config.epochs)

        # Predict Val & Test
        student_b_val_preds += trainer_b.predict(val_loader_b) / Config.num_folds
        student_b_test_preds += trainer_b.predict(test_loader_b) / Config.num_folds

        del (
            model_b,
            trainer_b,
            optimizer,
            scheduler,
            train_loader_b,
            val_loader_b,
            test_loader_b,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # 6. Evaluation & Analysis
    # -------------------------------------------------------------------------
    print("\n=== Evaluation ===")

    # Ensemble Predictions
    final_val_preds = (student_a_val_preds + student_b_val_preds) / 2.0
    final_test_preds = (student_a_test_preds + student_b_test_preds) / 2.0

    # Calculate Metric
    val_labels = val_df["Insult"].values
    final_auc = roc_auc_score(val_labels, final_val_preds)

    # Required Output Format
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    val_df["pred"] = final_val_preds
    val_df["error"] = np.abs(val_df["Insult"] - val_df["pred"])

    # Compute metadata features for correlation
    val_df["char_count"] = val_df["Comment"].fillna("").str.len()
    val_df["caps_ratio"] = (
        val_df["Comment"]
        .fillna("")
        .apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(1, len(str(x))))
    )

    correlations = val_df[["error", "char_count", "caps_ratio"]].corr()["error"]
    print("Error Correlations:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    threshold = 0.9603817733990148
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > threshold ({threshold}). Generating submission..."
        )

        sample_sub_path = os.path.join("./input", "sample_submission_null.csv")
        try:
            sub_df = pd.read_csv(sample_sub_path)
            sub_df["Insult"] = final_test_preds
            sub_df.to_csv(Config.submission_path, index=False)
            print(f"Submission saved to {Config.submission_path}")

        except Exception as e:
            print(f"Warning: Could not load sample submission file: {e}")
            print("Creating basic submission file...")
            out_df = test_df.copy()
            out_df["Insult"] = final_test_preds
            out_df.to_csv(Config.submission_path, index=False)
            print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nValidation metric ({final_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
