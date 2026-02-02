import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import warnings

# Import library modules
from library.config import Config
from library.utils import seed_everything, jaccard, calculate_consistency
from library.model import (
    TweetModel,
    TweetDataset,
    process_data,
    loss_fn,
    decode_prediction,
)
from library.engine import train_fn, predict_fn

# Suppress warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_pipeline():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for speed if necessary, but DeBERTa-Large fits in A100 easily.
    # We set epochs to 2 to ensure completion within 2 hours.
    EPOCHS = 2
    BATCH_SIZE = Config.TRAIN_BATCH_SIZE
    DEVICE = Config.DEVICE

    print(f"Device: {DEVICE}")

    # 2. Load Metadata
    df_train_full = pd.read_csv(Config.TRAIN_FILE)
    df_val_full = pd.read_csv(Config.VAL_FILE)
    df_test = pd.read_csv(Config.TEST_FILE)

    # 3. Preprocessing
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    # Filter neutrals for training (Idea Requirement)
    df_train_active = df_train_full[
        df_train_full["sentiment"] != "neutral"
    ].reset_index(drop=True)

    # Process Train Data (Stage 1)
    # We use cache_name="train_stage1"
    train_ids, train_masks, train_start, train_end, train_offsets, valid_indices = (
        process_data(
            df_train_active,
            tokenizer,
            Config.MAX_LEN,
            "train_stage1",
            load_cached_data=True,
            is_test=False,
        )
    )
    # Align dataframe with valid processed indices
    df_train_active = df_train_active.iloc[valid_indices].reset_index(drop=True)

    # Process Test Data
    test_ids, test_masks, _, _, test_offsets, _ = process_data(
        df_test, tokenizer, Config.MAX_LEN, "test", load_cached_data=True, is_test=True
    )

    test_dataset = TweetDataset(test_ids, test_masks)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # =========================================================================
    # Stage 1: Base Ensemble Training
    # =========================================================================
    print("Starting Stage 1: Base Ensemble Training...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    # Store test predictions for each fold: List of (start_probs, end_probs)
    stage1_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_active, df_train_active["sentiment"])
    ):
        # Create Datasets
        train_ds = TweetDataset(
            train_ids[train_idx],
            train_masks[train_idx],
            train_start[train_idx],
            train_end[train_idx],
        )
        # We don't strictly need val_ds for training loop if we just train for fixed epochs,
        # but we use it to save best model based on loss.
        val_ds = TweetDataset(
            train_ids[val_idx],
            train_masks[val_idx],
            train_start[val_idx],
            train_end[val_idx],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Init Model
        model = TweetModel()
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_train_steps = int(len(train_loader) * EPOCHS)
        num_warmup_steps = int(num_train_steps * Config.NUM_WARMUP_STEPS_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        best_loss = float("inf")
        best_model_path = os.path.join(
            Config.WORKING_DIR, f"model_stage1_fold_{fold}.bin"
        )

        # Training Loop
        for epoch in range(EPOCHS):
            _ = train_fn(train_loader, model, optimizer, DEVICE, scheduler)

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    start_labels = batch["start_labels"].to(DEVICE)
                    end_labels = batch["end_labels"].to(DEVICE)

                    s_logits, e_logits = model(input_ids, attention_mask)
                    loss = loss_fn(s_logits, e_logits, start_labels, end_labels)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)

        # Load best model and predict on Test
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        s_logits, e_logits = predict_fn(test_loader, model, DEVICE)
        # Convert logits to probs
        s_probs = np.exp(s_logits) / np.sum(np.exp(s_logits), axis=-1, keepdims=True)
        e_probs = np.exp(e_logits) / np.sum(np.exp(e_logits), axis=-1, keepdims=True)

        stage1_test_preds.append((s_probs, e_probs))

        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # Pseudo-Labeling
    # =========================================================================
    print("Generating Pseudo-Labels...")

    # Decode predictions for consistency check
    fold_str_preds = []
    for f in range(5):
        s_probs, e_probs = stage1_test_preds[f]
        preds = []
        for i in range(len(df_test)):
            text = str(df_test.loc[i, "text"])
            sentiment = df_test.loc[i, "sentiment"]
            offsets = test_offsets[i]
            pred_str = decode_prediction(
                s_probs[i], e_probs[i], text, offsets, sentiment
            )
            preds.append(pred_str)
        fold_str_preds.append(preds)

    consistency_scores = calculate_consistency(fold_str_preds)
    high_conf_indices = np.where(consistency_scores > Config.PSEUDO_LABEL_THRESHOLD)[0]

    # Create Augmented Dataset
    df_pseudo = df_test.iloc[high_conf_indices].copy()

    # Generate targets for pseudo labels (average probs)
    avg_s_probs = np.mean([x[0] for x in stage1_test_preds], axis=0)
    avg_e_probs = np.mean([x[1] for x in stage1_test_preds], axis=0)

    pseudo_texts = []
    for idx in high_conf_indices:
        text = str(df_test.loc[idx, "text"])
        sentiment = df_test.loc[idx, "sentiment"]
        offsets = test_offsets[idx]
        pred_str = decode_prediction(
            avg_s_probs[idx], avg_e_probs[idx], text, offsets, sentiment
        )
        pseudo_texts.append(pred_str)

    df_pseudo["selected_text"] = pseudo_texts

    # Filter neutrals from pseudo (should be handled by decode_prediction returning full text, but we exclude neutrals from training)
    df_pseudo = df_pseudo[df_pseudo["sentiment"] != "neutral"]

    df_train_augmented = pd.concat([df_train_active, df_pseudo], axis=0).reset_index(
        drop=True
    )

    # Process Augmented Data
    aug_ids, aug_masks, aug_start, aug_end, _, aug_valid = process_data(
        df_train_augmented,
        tokenizer,
        Config.MAX_LEN,
        "train_stage2",
        load_cached_data=False,
        is_test=False,
    )
    df_train_augmented = df_train_augmented.iloc[aug_valid].reset_index(drop=True)

    # =========================================================================
    # Stage 2: Augmented Ensemble Training
    # =========================================================================
    print("Starting Stage 2: Augmented Ensemble Training...")

    # Prepare Validation Data for Final Eval
    # We need to evaluate on the FULL validation set (including neutrals)
    # Process Val Data
    val_ids_full, val_masks_full, _, _, val_offsets_full, _ = process_data(
        df_val_full,
        tokenizer,
        Config.MAX_LEN,
        "val_full",
        load_cached_data=True,
        is_test=True,
    )
    val_dataset_full = TweetDataset(val_ids_full, val_masks_full)
    val_loader_full = DataLoader(
        val_dataset_full,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    skf_aug = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    # Accumulators for Ensemble Predictions
    final_val_s_logits = np.zeros((len(df_val_full), Config.MAX_LEN))
    final_val_e_logits = np.zeros((len(df_val_full), Config.MAX_LEN))

    final_test_s_logits = np.zeros((len(df_test), Config.MAX_LEN))
    final_test_e_logits = np.zeros((len(df_test), Config.MAX_LEN))

    for fold, (train_idx, val_idx) in enumerate(
        skf_aug.split(df_train_augmented, df_train_augmented["sentiment"])
    ):
        # Train/Val split on augmented data
        train_ds = TweetDataset(
            aug_ids[train_idx],
            aug_masks[train_idx],
            aug_start[train_idx],
            aug_end[train_idx],
        )
        val_ds = TweetDataset(
            aug_ids[val_idx], aug_masks[val_idx], aug_start[val_idx], aug_end[val_idx]
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        model = TweetModel()
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_train_steps = int(len(train_loader) * EPOCHS)
        num_warmup_steps = int(num_train_steps * Config.NUM_WARMUP_STEPS_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        best_loss = float("inf")
        best_model_path = os.path.join(
            Config.WORKING_DIR, f"model_stage2_fold_{fold}.bin"
        )

        for epoch in range(EPOCHS):
            _ = train_fn(train_loader, model, optimizer, DEVICE, scheduler)

            # Quick val check
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    start_labels = batch["start_labels"].to(DEVICE)
                    end_labels = batch["end_labels"].to(DEVICE)
                    s_logits, e_logits = model(input_ids, attention_mask)
                    loss = loss_fn(s_logits, e_logits, start_labels, end_labels)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        # Predict on Full Validation Set
        v_s_logits, v_e_logits = predict_fn(val_loader_full, model, DEVICE)
        final_val_s_logits += v_s_logits / 5.0
        final_val_e_logits += v_e_logits / 5.0

        # Predict on Test Set
        t_s_logits, t_e_logits = predict_fn(test_loader, model, DEVICE)
        final_test_s_logits += t_s_logits / 5.0
        final_test_e_logits += t_e_logits / 5.0

        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # Final Evaluation & Failure Analysis
    # =========================================================================

    # Convert logits to probs
    val_s_probs = np.exp(final_val_s_logits) / np.sum(
        np.exp(final_val_s_logits), axis=-1, keepdims=True
    )
    val_e_probs = np.exp(final_val_e_logits) / np.sum(
        np.exp(final_val_e_logits), axis=-1, keepdims=True
    )

    val_jaccards = []
    val_errors = []
    val_text_lens = []

    for i in range(len(df_val_full)):
        text = str(df_val_full.loc[i, "text"])
        sentiment = df_val_full.loc[i, "sentiment"]
        selected_text = str(df_val_full.loc[i, "selected_text"])
        offsets = val_offsets_full[i]

        pred_str = decode_prediction(
            val_s_probs[i], val_e_probs[i], text, offsets, sentiment
        )

        score = jaccard(selected_text, pred_str)
        val_jaccards.append(score)

        # Failure Analysis Data
        val_errors.append(1.0 - score)
        val_text_lens.append(len(text))

    final_metric = np.mean(val_jaccards)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    correlation = np.corrcoef(val_errors, val_text_lens)[0, 1]
    print(f"Correlation between Error and Text Length: {correlation:.4f}")

    # =========================================================================
    # Submission
    # =========================================================================

    if final_metric > 0.7164761348654044:
        print("Generating submission file...")

        test_s_probs = np.exp(final_test_s_logits) / np.sum(
            np.exp(final_test_s_logits), axis=-1, keepdims=True
        )
        test_e_probs = np.exp(final_test_e_logits) / np.sum(
            np.exp(final_test_e_logits), axis=-1, keepdims=True
        )

        submission_texts = []
        for i in range(len(df_test)):
            text = str(df_test.loc[i, "text"])
            sentiment = df_test.loc[i, "sentiment"]
            offsets = test_offsets[i]

            pred_str = decode_prediction(
                test_s_probs[i], test_e_probs[i], text, offsets, sentiment
            )

            # Ensure quoting is handled by pandas or manually
            # The requirement is quoted text. Pandas to_csv handles this with quoting settings.
            submission_texts.append(pred_str)

        sub_df = pd.DataFrame(
            {"textID": df_test["textID"], "selected_text": submission_texts}
        )

        # Force quoting for non-numeric (selected_text is string)
        # csv.QUOTE_NONNUMERIC (2) will quote strings. textID is string too.
        # The sample format: 2,"very good". '2' is ID.
        # If IDs are strings in dataframe, they will be quoted too: "2","very good".
        # If IDs are loaded as strings (which they are), they get quoted.
        # However, standard CSV readers handle quotes fine.
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    run_pipeline()
