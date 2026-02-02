import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import cfg
from library.utils import seed_everything, get_logger, get_score
from library.dataset import prepare_data, PearsonDataset, CollateFn
from library.model import CustomModel
from library.loss import CompositeLoss
from library.engine import get_optimizer_params, train_fn, valid_fn, AWP

# Suppress warnings
warnings.filterwarnings("ignore")


def run():
    # 1. Setup
    seed_everything(cfg.seed)
    logger = get_logger("run.log")

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    logger.info("Starting runfile execution...")
    logger.info(f"Device: {cfg.device}")

    # 2. Data Loading
    # Use cached data if available to save time
    train_df = prepare_data("train", load_cached_data=True)
    val_df = prepare_data("val", load_cached_data=True)
    test_df = prepare_data("test", load_cached_data=True)

    # For fast baseline, we can optionally subsample if debug is True
    # (Though config.debug is False by default, this handles the logic)
    if cfg.debug:
        logger.info("Debug mode: Subsampling data...")
        train_df = train_df.sample(
            n=min(len(train_df), 1000), random_state=cfg.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), 500), random_state=cfg.seed
        ).reset_index(drop=True)

    # 3. Tokenizer & Datasets
    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset = PearsonDataset(train_df, tokenizer, max_len=cfg.max_len)
    val_dataset = PearsonDataset(val_df, tokenizer, max_len=cfg.max_len)
    test_dataset = PearsonDataset(test_df, tokenizer, max_len=cfg.max_len)

    collate_fn = CollateFn(tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 4. Model Initialization
    logger.info("Initializing model...")
    model = CustomModel()
    model.to(cfg.device)

    # 5. Training Setup
    # Optimizer with LLRD
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=cfg.learning_rate,
        decoder_lr=cfg.learning_rate * 5,  # Higher LR for head
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=cfg.learning_rate, eps=cfg.eps, betas=cfg.betas
    )

    # Scheduler
    num_training_steps = len(train_loader) * cfg.epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss
    criterion = CompositeLoss()

    # AWP
    awp = None
    if cfg.use_awp:
        awp = AWP(model, optimizer, adv_lr=cfg.awp_lr, adv_eps=cfg.awp_eps)

    # 6. Training Loop
    best_score = -1.0

    logger.info(f"Starting training for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        # Train
        avg_loss = train_fn(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            cfg.device,
            awp,
        )

        # Validate
        val_loss, val_score, _ = valid_fn(val_loader, model, criterion, cfg.device)

        logger.info(
            f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Score: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            logger.info(f"New best score: {best_score:.4f}. Saving model...")
            torch.save(model.state_dict(), cfg.model_output_path)

    # 7. Final Validation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(cfg.model_output_path, map_location=cfg.device))

    # Run inference on validation set
    _, final_val_score, val_preds = valid_fn(val_loader, model, criterion, cfg.device)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_val_score}")

    # Failure Analysis
    logger.info("Performing failure analysis...")
    val_df["pred"] = val_preds
    val_df["abs_error"] = (val_df["score"] - val_df["pred"]).abs()

    # Feature extraction for analysis
    val_df["anchor_len"] = val_df["anchor"].astype(str).apply(len)
    val_df["target_len"] = val_df["target"].astype(str).apply(len)
    # Context frequency
    context_counts = val_df["context"].value_counts().to_dict()
    val_df["context_freq"] = val_df["context"].map(context_counts)

    # Correlation of error with features
    features_to_analyze = ["anchor_len", "target_len", "context_freq"]
    logger.info("Correlation between Absolute Error and Features:")
    for feat in features_to_analyze:
        if feat in val_df.columns:
            corr = val_df["abs_error"].corr(val_df[feat])
            logger.info(f"  {feat}: {corr:.4f}")
            print(f"Failure Analysis - Correlation ({feat} vs Error): {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.8582661747932434

    if final_val_score > THRESHOLD:
        logger.info(
            f"Validation score ({final_val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        model.eval()
        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(cfg.device)

                with torch.amp.autocast("cuda"):
                    outputs = model(batch["input_ids"], batch["attention_mask"])

                preds = outputs["logits"].view(-1).cpu().numpy()
                test_preds.append(preds)

        test_predictions = np.concatenate(test_preds)

        # Post-processing: Clip to [0, 1]
        test_predictions = np.clip(test_predictions, 0, 1)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_df["id"], "score": test_predictions})

        # Save
        submission_path = os.path.join(submission_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.warning(
            f"Validation score ({final_val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
