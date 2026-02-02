import os
import sys
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import CFG
from library.utils import seed_everything, get_score
from library.data import process_data, get_cpc_texts, PhraseDataset
from library.model import CustomDeberta
from library.engine import train_fn, valid_fn, inference_fn
from library.stacking import train_stacking_model


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for this run
    CFG.output_dir = "./working"
    CFG.epochs = 2  # Limit epochs to ensure completion within 2 hours
    CFG.awp_start_epoch = 1.0  # Start AWP at the 2nd epoch (index 1)

    # Create output directory
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Set Random Seed
    seed_everything(CFG.seed)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading & Preprocessing
    # ==========================================
    print("Loading and preprocessing data...")
    cpc_map = get_cpc_texts()

    # Load metadata files
    # We use the metadata files directly as they contain the text
    meta_train_path = "./metadata/train.csv"
    meta_val_path = "./metadata/val.csv"
    meta_test_path = "./metadata/test.csv"

    train_meta = pd.read_csv(meta_train_path)
    val_meta = pd.read_csv(meta_val_path)
    test_df = pd.read_csv(meta_test_path)

    # Combine metadata train and val to create the full training set for 5-Fold CV
    # We will use the original 'val.csv' (val_meta) as the pure hold-out set for final evaluation
    train_full = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)
    val_holdout = val_meta.copy()  # Keep a separate copy for hold-out evaluation

    # Preprocess (add structural features and context text)
    train_full = process_data(train_full, cpc_map)
    val_holdout = process_data(val_holdout, cpc_map)
    test_df = process_data(test_df, cpc_map)

    print(f"Full Training Data Shape: {train_full.shape}")
    print(f"Hold-out Validation Shape: {val_holdout.shape}")
    print(f"Test Data Shape: {test_df.shape}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # ==========================================
    # 3. Level 1: 5-Fold CV Training
    # ==========================================
    # We perform Stratified K-Fold on the full training data
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    # Arrays to store predictions
    # OOF probabilities for the training set (to train Level 2)
    oof_probs = np.zeros((len(train_full), CFG.num_classes))

    # Accumulated probabilities for Hold-out Val and Test sets (averaged across folds)
    val_probs_sum = np.zeros((len(val_holdout), CFG.num_classes))
    test_probs_sum = np.zeros((len(test_df), CFG.num_classes))

    # Stratification target (discretized scores)
    y_stratify = (train_full["score"] * 4).round().astype(int)

    print(f"\nStarting Level 1 Training: {CFG.n_fold} Folds, {CFG.epochs} Epochs each.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_full, y_stratify)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        df_train = train_full.iloc[train_idx].reset_index(drop=True)
        df_val = train_full.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_ds = PhraseDataset(df_train, tokenizer, CFG.max_len, mode="train")
        val_ds = PhraseDataset(df_val, tokenizer, CFG.max_len, mode="val")

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = CustomDeberta(CFG.model_name)
        model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.encoder_lr,
            weight_decay=CFG.weight_decay,
            eps=CFG.eps,
        )

        # Scheduler
        num_train_steps = int(
            len(df_train)
            / CFG.batch_size
            / CFG.gradient_accumulation_steps
            * CFG.epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss Function
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

        # Training Loop
        best_score = -1.0
        best_model_path = os.path.join(CFG.output_dir, f"model_fold_{fold}.bin")

        for epoch in range(CFG.epochs):
            start_time = pd.Timestamp.now()

            # Train
            avg_loss = train_fn(
                fold,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
            )

            # Validate
            val_loss, val_score, _ = valid_fn(val_loader, model, criterion, device)

            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).seconds

            print(
                f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pearson: {val_score:.4f} | Time: {duration}s"
            )

            # Save Best Model
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best Pearson: {best_score:.4f}")

        # Load Best Model for Inference
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # 1. Inference on OOF (Validation Fold)
        _, _, fold_oof_probs = valid_fn(val_loader, model, criterion, device)
        oof_probs[val_idx] = fold_oof_probs

        # 2. Inference on Hold-out Validation Set
        # Use larger batch size for inference
        val_holdout_ds = PhraseDataset(val_holdout, tokenizer, CFG.max_len, mode="test")
        val_holdout_loader = DataLoader(
            val_holdout_ds,
            batch_size=CFG.batch_size * 2,
            shuffle=False,
            num_workers=CFG.num_workers,
        )
        fold_val_probs = inference_fn(val_holdout_loader, model, device)
        val_probs_sum += fold_val_probs

        # 3. Inference on Test Set
        test_ds = PhraseDataset(test_df, tokenizer, CFG.max_len, mode="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=CFG.batch_size * 2,
            shuffle=False,
            num_workers=CFG.num_workers,
        )
        fold_test_probs = inference_fn(test_loader, model, device)
        test_probs_sum += fold_test_probs

        # Cleanup to free memory
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

    # Average predictions across folds
    val_probs_avg = val_probs_sum / CFG.n_fold
    test_probs_avg = test_probs_sum / CFG.n_fold

    # ==========================================
    # 4. Level 2: Stacking & Validation
    # ==========================================
    print("\n=== Level 2 Stacking & Evaluation ===")

    # A. Evaluate on Hold-out Validation Set
    # We train the stacker on train_full (using OOF preds) and predict on val_holdout

    # Temporarily redirect submission path to avoid overwriting the final submission
    original_sub_path = CFG.submission_path
    val_pred_path = os.path.join(CFG.output_dir, "val_predictions.csv")
    CFG.submission_path = val_pred_path

    print("Training Stacker and predicting on Hold-out Validation Set...")
    # Note: load_cached_data=False ensures we compute features for the specific test_df passed (val_holdout)
    val_preds = train_stacking_model(
        train_df=train_full,
        test_df=val_holdout,
        oof_probs=oof_probs,
        test_probs=val_probs_avg,
        load_cached_data=False,
    )

    # Calculate Final Validation Metric
    final_val_score = get_score(val_holdout["score"].values, val_preds)
    print(f"Final Validation Metric: {final_val_score:.16f}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_holdout["score"].values - val_preds)

    # Calculate correlation with structural features
    features_to_analyze = ["feat_lev", "feat_jac", "feat_len"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features_to_analyze:
        if feat in val_holdout.columns:
            corr = np.corrcoef(errors, val_holdout[feat])[0, 1]
            print(f"  Error vs {feat}: {corr:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = 0.8654320295612139

    if final_val_score > threshold:
        print(
            f"\nValidation score ({final_val_score:.4f}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Restore original submission path
        CFG.submission_path = original_sub_path

        # Retrain Stacker (fast) and predict on Test Set
        _ = train_stacking_model(
            train_df=train_full,
            test_df=test_df,
            oof_probs=oof_probs,
            test_probs=test_probs_avg,
            load_cached_data=False,
        )
        print("Submission generation complete.")

    else:
        print(
            f"\nValidation score ({final_val_score:.4f}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
