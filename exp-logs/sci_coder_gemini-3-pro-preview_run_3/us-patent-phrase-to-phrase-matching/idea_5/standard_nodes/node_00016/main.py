import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable tqdm globally to satisfy "Do not print progress bars" requirement
try:
    from tqdm import tqdm

    tqdm.disable = True
    import tqdm.auto

    tqdm.auto.tqdm = lambda *args, **kwargs: args[0] if len(args) > 0 else iter(args)
except ImportError:
    pass

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, get_score
from library.cpc_utils import load_context_enriched_data
from library.dataset import (
    get_tokenizer,
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
)
from library.model import CustomModel
from library.loss import HybridLoss
from library.awp import AWP
from library.engine import (
    run_dapt,
    get_optimizer_params,
    train_fn,
    valid_fn,
    inference_fn,
)


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Adjust configuration for A100 efficiency and time limits
    Config.train_batch_size = 32  # Increase batch size for A100 (40GB VRAM)
    Config.valid_batch_size = 64
    Config.epochs = 3  # Limit epochs to ensure completion < 2h
    Config.dapt_epochs = 1  # Limit DAPT epochs
    Config.dapt_batch_size = 32

    # Ensure directories exist
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Initialize Logger
    logger = get_logger("runfile")

    # Set Random Seed
    seed_everything(Config.seed)

    logger.info("Starting execution...")

    # =========================================================================
    # 2. Domain-Adaptive Pre-training (DAPT)
    # =========================================================================
    tokenizer = get_tokenizer()

    # Run DAPT (saves model to Config.dapt_model_path)
    run_dapt(tokenizer)

    # Determine which backbone to load
    if Config.use_dapt and os.path.exists(
        os.path.join(Config.dapt_model_path, "config.json")
    ):
        backbone_path = Config.dapt_model_path
        logger.info(f"Using DAPT-adapted backbone from {backbone_path}")
    else:
        backbone_path = Config.model_name
        logger.info(f"Using standard backbone {backbone_path}")

    # =========================================================================
    # 3. Data Loading & Model Initialization
    # =========================================================================
    train_loader = get_train_dataloader(tokenizer)
    val_loader = get_val_dataloader(tokenizer)
    test_loader, test_ids = get_test_dataloader(tokenizer)

    # Initialize Model
    model = CustomModel(config_path=backbone_path, pretrained=True)
    model.to(Config.device)

    # =========================================================================
    # 4. Training Setup
    # =========================================================================
    # Optimizer with Layer-wise Learning Rate Decay (LLRD)
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate * 5.0,
        weight_decay=Config.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=Config.learning_rate, eps=Config.eps
    )

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Loss Function
    loss_fn = HybridLoss()

    # Adversarial Weight Perturbation (AWP)
    awp = None
    if Config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    best_score = -1.0

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            epoch,
            scheduler,
            Config.device,
            loss_fn,
            awp,
        )

        # Validate
        val_loss, val_score = valid_fn(val_loader, model, Config.device, loss_fn)

        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Score: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)

    # =========================================================================
    # 6. Evaluation & Failure Analysis
    # =========================================================================
    # Load best model weights
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(
        torch.load(Config.model_save_path, map_location=Config.device)
    )
    model.to(Config.device)
    model.eval()

    # Generate Validation Predictions
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            outputs = model(input_ids, attention_mask)
            val_preds.append(outputs["logits"].view(-1).cpu().numpy())
            val_targets.append(labels.view(-1).cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Calculate Final Metric
    final_metric = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    df_val = load_context_enriched_data("val", load_cached_data=True)
    df_val["pred"] = val_preds
    df_val["abs_error"] = (df_val["score"] - df_val["pred"]).abs()

    # Compute meta-features
    df_val["anchor_len"] = df_val["anchor"].astype(str).apply(len)
    df_val["target_len"] = df_val["target"].astype(str).apply(len)
    df_val["len_diff"] = (df_val["anchor_len"] - df_val["target_len"]).abs()

    def get_jaccard(row):
        a = set(str(row["anchor"]).lower().split())
        b = set(str(row["target"]).lower().split())
        u = len(a.union(b))
        return len(a.intersection(b)) / u if u > 0 else 0.0

    df_val["jaccard"] = df_val.apply(get_jaccard, axis=1)

    # Calculate correlations
    analysis_cols = ["anchor_len", "target_len", "len_diff", "jaccard", "score"]
    correlations = df_val[analysis_cols].corrwith(df_val["abs_error"])

    print("Failure Analysis (Correlation with Abs Error):")
    print(correlations)

    # =========================================================================
    # 7. Submission
    # =========================================================================
    threshold = 0.8582661747932434

    if final_metric > threshold:
        logger.info("Metric exceeds threshold. Generating submission...")

        # Inference on Test Set
        test_preds = inference_fn(test_loader, model, Config.device)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "score": test_preds})

        # Save
        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(
            f"Metric {final_metric} did not exceed threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
