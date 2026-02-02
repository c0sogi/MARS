import os
import sys
import gc
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_pearson_score, get_cpc_texts
from library.data import PhraseDataset
from library.model import CustomDeberta
from library.awp import AWP
from library.train import train_fn, valid_fn


def main():
    # --- 1. Configuration & Setup ---
    seed_everything(Config.seed)

    # Override Config for Fast Baseline on A100
    Config.epochs = 1  # 1 Epoch is sufficient for large models on this size
    Config.train_batch_size = 16  # Increase batch size for speed (A100 40GB)
    Config.valid_batch_size = 32
    Config.awp_start_epoch = 0  # Apply AWP from the start since we only have 1 epoch
    Config.num_workers = 4

    print(
        f"Configuration: Epochs={Config.epochs}, Batch={Config.train_batch_size}, AWP Start={Config.awp_start_epoch}"
    )

    device = Config.device

    # --- 2. Data Loading & Preprocessing ---
    print("Loading data...")
    # Load Train and Val from metadata to combine for K-Fold
    df_train_meta = pd.read_csv(Config.train_path)
    df_val_meta = pd.read_csv(Config.val_path)
    full_df = pd.concat([df_train_meta, df_val_meta]).reset_index(drop=True)

    df_test = pd.read_csv(Config.test_path)

    # Context Enrichment
    cpc_texts = get_cpc_texts(Config.cpc_codes_path)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    def prepare_input_text(df):
        def construct_text(row):
            code = row.get("context", "")
            context_desc = cpc_texts.get(code, code)
            anchor = row.get("anchor", "")
            target = row.get("target", "")
            # Format: Context [SEP] Anchor [SEP] Target
            return f"{str(context_desc).strip()}{tokenizer.sep_token}{str(anchor).strip()}{tokenizer.sep_token}{str(target).strip()}"

        df["input_text"] = df.apply(construct_text, axis=1)
        return df

    full_df = prepare_input_text(full_df)
    df_test = prepare_input_text(df_test)

    print(f"Full Training Data: {full_df.shape}")
    print(f"Test Data: {df_test.shape}")

    # --- 3. Stratified K-Fold Training ---
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    oof_preds = np.zeros(len(full_df))
    test_preds_folds = []

    # Stratify by score (convert to string to treat as categorical)
    y_stratify = full_df["score"].astype(str)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y_stratify)):
        print(f"\n=== Starting Fold {fold+1}/{Config.n_folds} ===")

        # Prepare DataLoaders
        train_fold = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold = full_df.iloc[val_idx].reset_index(drop=True)

        train_ds = PhraseDataset(
            train_fold, tokenizer, Config.max_length, is_test=False
        )
        val_ds = PhraseDataset(val_fold, tokenizer, Config.max_length, is_test=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = CustomDeberta(Config.model_name, pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_parameters,
            lr=Config.learning_rate,
            eps=Config.eps,
            betas=Config.betas,
        )

        num_train_steps = len(train_loader) * Config.epochs
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Loss, Scaler, AWP
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler = GradScaler()
        awp = AWP(model, optimizer, scaler=scaler) if Config.use_awp else None

        # Training Loop
        best_pearson = -1.0
        model_save_path = os.path.join(Config.output_dir, f"model_fold_{fold}.bin")

        for epoch in range(Config.epochs):
            train_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                awp,
                scaler,
            )
            val_loss, val_preds, val_labels = valid_fn(
                val_loader, model, criterion, device
            )
            pearson = compute_pearson_score(val_labels, val_preds)

            print(
                f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Pearson: {pearson:.5f}"
            )

            if pearson > best_pearson:
                best_pearson = pearson
                torch.save(model.state_dict(), model_save_path)

        # --- Inference on Fold ---
        # Load Best Model
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        # OOF Predictions
        _, val_preds, _ = valid_fn(val_loader, model, criterion, device)
        oof_preds[val_idx] = val_preds

        # Test Predictions
        test_ds = PhraseDataset(df_test, tokenizer, Config.max_length, is_test=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        fold_test_preds = []
        class_values = torch.tensor([0.0, 0.25, 0.50, 0.75, 1.00], device=device)

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(enabled=True):
                    logits = model(input_ids, attention_mask)

                probs = torch.softmax(logits, dim=1)
                batch_preds = torch.sum(probs * class_values, dim=1)
                fold_test_preds.append(batch_preds.cpu().numpy())

        test_preds_folds.append(np.concatenate(fold_test_preds))

        # Cleanup
        del model, optimizer, scheduler, scaler, awp, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # --- 4. Final Validation & Analysis ---
    final_metric = compute_pearson_score(full_df["score"].values, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    full_df["pred"] = oof_preds
    full_df["error"] = (full_df["score"] - full_df["pred"]).abs()

    # Feature Engineering for Analysis
    full_df["anchor_len"] = full_df["anchor"].astype(str).apply(len)
    full_df["target_len"] = full_df["target"].astype(str).apply(len)

    def calculate_jaccard(row):
        set_a = set(str(row["anchor"]).lower().split())
        set_b = set(str(row["target"]).lower().split())
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    full_df["jaccard"] = full_df.apply(calculate_jaccard, axis=1)

    # Correlations
    corr_anchor = full_df["error"].corr(full_df["anchor_len"])
    corr_target = full_df["error"].corr(full_df["target_len"])
    corr_jaccard = full_df["error"].corr(full_df["jaccard"])

    print("Correlation of Error Magnitude with Features:")
    print(f"  Anchor Length: {corr_anchor:.6f}")
    print(f"  Target Length: {corr_target:.6f}")
    print(f"  Jaccard Similarity: {corr_jaccard:.6f}")

    # --- 5. Submission ---
    THRESHOLD = 0.8550264305718601
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        avg_test_preds = np.mean(test_preds_folds, axis=0)

        submission = pd.DataFrame({"id": df_test["id"], "score": avg_test_preds})

        os.makedirs("./submission", exist_ok=True)
        submission.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
