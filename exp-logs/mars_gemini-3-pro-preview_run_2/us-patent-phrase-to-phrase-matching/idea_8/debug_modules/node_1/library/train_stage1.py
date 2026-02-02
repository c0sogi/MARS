import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.dataset import load_and_process_data, PhraseDataset
from library.model import DebertaV3WithFeatures
from library.engine import train_loop


def predict(model, dataloader, device):
    """
    Generates predictions (expected values) for a given dataloader.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader containing the data.
        device: 'cuda' or 'cpu'.

    Returns:
        ids: List of IDs.
        preds: List of predicted scores (expected values).
    """
    model.eval()
    preds = []
    ids = []

    # Mapping class indices 0..4 to scores 0.0..1.0
    score_map = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            structural_features = data["structural_features"].to(device)
            batch_ids = data["id"]

            # Forward pass
            outputs = model(input_ids, attention_mask, structural_features)

            # Softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Calculate Expected Value: sum(prob_i * score_i)
            expected_scores = torch.sum(probs * score_map, dim=1)

            preds.extend(expected_scores.cpu().numpy())
            ids.extend(batch_ids)

    return ids, preds


def run_kfold(debug=False, epochs=Config.EPOCHS):
    """
    Orchestrates the Stratified K-Fold training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data for quick testing.
        epochs (int): Number of training epochs per fold.
    """
    seed_everything(Config.SEED)

    print(f"Starting Stage 1 Training (Debug={debug})...")

    # 1. Load and Process Data
    # We combine the metadata train and val splits to perform our own K-Fold
    print("Loading and processing datasets...")
    train_data = load_and_process_data("train", debug=debug)
    val_data = load_and_process_data("val", debug=debug)
    test_data = load_and_process_data("test", debug=debug)

    # Combine train and val for cross-validation
    full_train_df = pd.concat([train_data, val_data]).reset_index(drop=True)

    print(f"Full Training Data Shape: {full_train_df.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    # 2. Prepare Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. Prepare Test Loader (Common for all folds)
    test_dataset = PhraseDataset(
        test_data, tokenizer, max_length=Config.MAX_LENGTH, inference_only=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Containers for results
    oof_preds = []
    oof_ids = []
    oof_targets = []
    oof_folds = []

    # Array to accumulate test predictions from each fold
    test_preds_accumulator = np.zeros(len(test_data))

    # 4. Stratified K-Fold
    # Stratify by 'score' to ensure balanced label distribution
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    # We bin continuous scores for stratification if necessary, but here scores are discrete
    y_stratify = full_train_df["score"].astype(str).values

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_df, y_stratify)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.FOLDS} {'='*20}")

        # Split Data
        df_train_fold = full_train_df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = full_train_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_ds = PhraseDataset(df_train_fold, tokenizer, max_length=Config.MAX_LENGTH)
        val_ds = PhraseDataset(df_val_fold, tokenizer, max_length=Config.MAX_LENGTH)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        # num_features=6 based on features.py (lev_dist, lev_norm, jaccard, len_diff, len_ratio, word_diff)
        model = DebertaV3WithFeatures(num_features=6, num_classes=5)
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        num_training_steps = int(len(train_loader) / Config.GRAD_ACCUM_STEPS * epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.NUM_WARMUP_STEPS,
            num_training_steps=num_training_steps,
        )

        # Train Loop
        model_save_path = os.path.join(Config.MODELS_DIR, f"model_fold_{fold}.bin")
        train_loop(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            Config.DEVICE,
            epochs,
            model_save_path,
        )

        # Load Best Model for Inference
        print(f"Loading best model for Fold {fold} inference...")
        model.load_state_dict(torch.load(model_save_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        # OOF Inference
        val_ids, val_preds = predict(model, val_loader, Config.DEVICE)

        oof_ids.extend(val_ids)
        oof_preds.extend(val_preds)
        oof_targets.extend(df_val_fold["score"].values)
        oof_folds.extend([fold] * len(df_val_fold))

        # Test Inference
        _, fold_test_preds = predict(model, test_loader, Config.DEVICE)
        test_preds_accumulator += np.array(fold_test_preds)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Process Results
    print("\nProcessing final results...")

    # Save OOF Predictions
    oof_df = pd.DataFrame(
        {"id": oof_ids, "pred": oof_preds, "score": oof_targets, "fold": oof_folds}
    )
    oof_path = os.path.join(Config.WORKING_DIR, "stage1_oof.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to {oof_path}")

    # Calculate OOF Score
    from library.utils import compute_score

    oof_score = compute_score(oof_df["score"].values, oof_df["pred"].values)
    print(f"Overall OOF Pearson Score: {oof_score}")

    # Average Test Predictions
    avg_test_preds = test_preds_accumulator / Config.FOLDS

    # Save Test Predictions (Intermediate for Stage 2)
    test_pred_df = pd.DataFrame({"id": test_data["id"].values, "pred": avg_test_preds})
    test_pred_path = os.path.join(Config.WORKING_DIR, "stage1_test.csv")
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved Stage 1 Test predictions to {test_pred_path}")

    # Save Baseline Submission (Required format)
    submission_df = test_pred_df.rename(columns={"pred": "score"})
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Saved baseline submission to {Config.SUBMISSION_FILE}")

    print("Stage 1 Training Complete.")
