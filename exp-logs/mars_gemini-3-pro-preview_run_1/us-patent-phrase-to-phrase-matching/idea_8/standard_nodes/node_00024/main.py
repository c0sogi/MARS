import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import CFG
from library.utils import seed_everything, get_logger, get_score, get_device
from library.data import preprocess_data, prepare_loaders, prepare_inference_loader
from library.model import CustomModel, get_optimizer_params
from library.engine import train_fn, valid_fn, inference_fn, ModelEMA

# Suppress warnings and logs
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def main():
    # 1. Configuration & Setup
    seed_everything(CFG.seed)
    device = get_device()

    # Override CFG for speed and optimization within 2 hours
    CFG.batch_size = 32  # Increase batch size for A100
    CFG.num_epochs = 2  # Reduce epochs to fit time limit
    CFG.working_dir = "./working/idea_8_run"
    os.makedirs(CFG.working_dir, exist_ok=True)

    # Setup Logger
    logger = get_logger(os.path.join(CFG.working_dir, "train.log"))
    logger.info(f"Starting training with device: {device}")

    # 2. Data Preprocessing
    # This will load metadata, merge CPC texts, and create folds
    train_df, test_df = preprocess_data(CFG, load_cached_data=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # Container for OOF predictions
    # We will store predictions mapped by ID to reconstruct the validation set later
    oof_preds_dict = {}

    # 3. Training Loop (Stratified Group K-Fold)
    for fold in range(CFG.n_folds):
        logger.info(f"=== Starting Fold {fold} ===")

        # Prepare Loaders
        train_loader, val_loader = prepare_loaders(fold, tokenizer, CFG)

        # Initialize Model
        model = CustomModel(CFG, pretrained=True)
        model.to(device)

        # Initialize EMA
        model_ema = ModelEMA(model, decay=CFG.ema_decay) if CFG.use_ema else None

        # Optimizer & Scheduler
        optimizer_parameters = get_optimizer_params(
            model,
            encoder_lr=CFG.learning_rate,
            decoder_lr=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
        )
        optimizer = AdamW(
            optimizer_parameters, lr=CFG.learning_rate, eps=CFG.eps, betas=CFG.betas
        )

        num_train_steps = int(len(train_loader) * CFG.num_epochs / CFG.accum_iter)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss Function
        criterion = nn.MSELoss()

        best_score = -1
        best_loss = np.inf

        # Epoch Loop
        for epoch in range(CFG.num_epochs):
            # Train
            avg_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                CFG,
                model_ema,
            )

            # Validate
            val_loss, val_score = valid_fn(
                val_loader, model, criterion, device, CFG, model_ema
            )

            logger.info(
                f"Fold {fold} | Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}"
            )

            # Save Best Model (based on Score)
            if val_score > best_score:
                best_score = val_score
                # Save EMA weights if available, else model weights
                if model_ema is not None:
                    model_ema.apply_shadow()
                    torch.save(
                        model.state_dict(),
                        os.path.join(CFG.working_dir, f"model_fold_{fold}.pth"),
                    )
                    model_ema.restore()
                else:
                    torch.save(
                        model.state_dict(),
                        os.path.join(CFG.working_dir, f"model_fold_{fold}.pth"),
                    )

        # Generate OOF predictions for this fold using the best model
        # Reload best model
        model.load_state_dict(
            torch.load(os.path.join(CFG.working_dir, f"model_fold_{fold}.pth"))
        )
        model.to(device)
        if model_ema:
            # If we saved EMA weights directly to the file, we just load them.
            # The logic above saves the shadow weights into the file.
            pass

        # Inference on Validation Set
        val_preds = inference_fn(
            val_loader, model, device, model_ema=None
        )  # EMA already applied/saved

        # Map predictions to IDs
        val_ids = val_loader.dataset.ids
        for i, uid in enumerate(val_ids):
            oof_preds_dict[uid] = val_preds[i]

        # Clean up to save memory
        del model, optimizer, scheduler, train_loader, val_loader, model_ema
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    logger.info("=== Validation Assessment ===")

    # Load the specific hold-out validation set metadata
    val_meta_df = pd.read_csv(CFG.val_file)

    # Align OOF predictions with the validation metadata
    # We filter our OOF dict to only include IDs present in val.csv
    val_preds_aligned = []
    val_targets_aligned = []
    val_ids_aligned = []

    missing_ids = 0
    for idx, row in val_meta_df.iterrows():
        uid = row["id"]
        if uid in oof_preds_dict:
            val_preds_aligned.append(oof_preds_dict[uid])
            val_targets_aligned.append(row["score"])
            val_ids_aligned.append(uid)
        else:
            missing_ids += 1

    if missing_ids > 0:
        logger.warning(
            f"Missing predictions for {missing_ids} items in validation set."
        )

    val_preds_aligned = np.array(val_preds_aligned)
    val_targets_aligned = np.array(val_targets_aligned)

    # Compute Metric
    final_metric = get_score(val_targets_aligned, val_preds_aligned)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("=== Failure Analysis ===")

    # Create Analysis DataFrame
    analysis_df = val_meta_df[val_meta_df["id"].isin(val_ids_aligned)].copy()
    analysis_df["pred"] = val_preds_aligned
    analysis_df["error"] = np.abs(analysis_df["score"] - analysis_df["pred"])

    # Feature Engineering for Analysis
    analysis_df["anchor_len"] = analysis_df["anchor"].astype(str).apply(len)
    analysis_df["target_len"] = analysis_df["target"].astype(str).apply(len)
    analysis_df["context_len"] = (
        analysis_df["context"].astype(str).apply(len)
    )  # Length of code, not desc

    # Correlations
    correlations = {}
    for col in ["anchor_len", "target_len", "score"]:
        corr, _ = pearsonr(analysis_df["error"], analysis_df[col])
        correlations[col] = corr

    print("Correlation between Error Magnitude and Features:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.8673

    if final_metric > THRESHOLD:
        logger.info(
            f"Metric {final_metric:.4f} > {THRESHOLD}. Generating submission..."
        )

        test_loader = prepare_inference_loader(tokenizer, CFG)
        test_ids = test_loader.dataset.ids

        fold_preds = []

        for fold in range(CFG.n_folds):
            model_path = os.path.join(CFG.working_dir, f"model_fold_{fold}.pth")

            # Load Model
            model = CustomModel(CFG, pretrained=False)
            model.load_state_dict(torch.load(model_path))
            model.to(device)

            # Predict
            preds = inference_fn(test_loader, model, device)
            fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        # Average Predictions
        avg_preds = np.mean(fold_preds, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "score": avg_preds})

        # Clip scores to valid range [0, 1] (optional but good practice)
        submission_df["score"] = submission_df["score"].clip(0, 1)

        # Save
        submission_path = os.path.join(CFG.submission_dir, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.info(
            f"Metric {final_metric:.4f} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
