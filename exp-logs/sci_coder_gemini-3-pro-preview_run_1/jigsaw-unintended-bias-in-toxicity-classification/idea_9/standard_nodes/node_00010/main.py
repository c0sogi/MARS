import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Import library components
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.data_processing import (
    get_data,
    get_loaders,
    get_test_loader,
    get_tokenizer,
)
from library.model import JigsawModel
from library.losses import HybridLoss
from library.awp import AWP
from library.trainer import (
    train_fn,
    valid_fn,
    inference_fn,
    compute_bias_metrics,
    run_dapt,
)

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Enable debug mode and reduce epochs/samples to ensure execution within 2 hours.
CFG.debug = True
CFG.debug_sample_size = 5000  # Small subset for fast execution
CFG.epochs = 1  # Single epoch for baseline
CFG.mlm_epochs = 1  # Quick DAPT
CFG.awp_start_epoch = 0  # Apply AWP immediately since we only have 1 epoch
CFG.print_freq = 50

# Ensure output directories exist
os.makedirs(CFG.output_dir, exist_ok=True)
os.makedirs(CFG.submission_dir, exist_ok=True)

logger = get_logger()


def failure_analysis(val_df, preds):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    logger.info("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude
    # Target is continuous in val_df, preds are continuous probabilities
    # We use absolute error
    val_df = val_df.copy()
    val_df["pred"] = preds
    val_df["error"] = (val_df["target"] - val_df["pred"]).abs()

    # Feature Engineering for Analysis
    val_df["comment_len"] = val_df["comment_text"].astype(str).apply(len)

    # Select features to correlate
    features = ["target", "comment_len"] + CFG.identity_cols

    # Compute correlations
    correlations = {}
    for feat in features:
        if feat in val_df.columns:
            # Handle NaNs in identities by filling with 0 for analysis
            series = val_df[feat].fillna(0.0)
            if series.std() > 0:  # Avoid constant columns
                corr = series.corr(val_df["error"])
                correlations[feat] = corr
            else:
                correlations[feat] = 0.0

    # Print results
    logger.info("-" * 40)
    logger.info("Correlation between Error Magnitude and Features:")
    logger.info("-" * 40)
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs:
        logger.info(f"{feat: <30}: {corr:.6f}")
    logger.info("-" * 40)


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    logger.info(f"Starting Fast Baseline Run (Debug={CFG.debug}, Epochs={CFG.epochs})")

    # 2. Data Loading
    tokenizer = get_tokenizer()
    # Load data (cached if available, subsampled due to debug=True)
    train_df, val_df, test_df = get_data(
        load_cached_data=True, debug=CFG.debug, debug_size=CFG.debug_sample_size
    )

    # 3. Domain Adaptive Pretraining (DAPT)
    # Run briefly on the (debug) dataset
    backbone_path = os.path.join(CFG.cache_dir, "dapt_backbone.pth")
    if not os.path.exists(backbone_path):
        run_dapt(train_df, test_df, tokenizer)
    else:
        logger.info("DAPT backbone found in cache.")

    # 4. Prepare Loaders
    train_loader, val_loader = get_loaders(train_df, val_df, tokenizer)

    # 5. Model Initialization
    model = JigsawModel(pretrained=True)

    # Load DAPT weights if available
    if os.path.exists(backbone_path):
        logger.info(f"Loading DAPT weights from {backbone_path}")
        state_dict = torch.load(backbone_path, map_location="cpu")
        model.model.load_state_dict(state_dict, strict=False)

    model.to(CFG.device)

    # 6. Optimization Setup
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "lr": CFG.encoder_lr,
            "weight_decay": CFG.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "lr": CFG.encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "model" not in n],
            "lr": CFG.decoder_lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, eps=CFG.eps, betas=CFG.betas)

    num_train_steps = int(len(train_loader) * CFG.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
        num_training_steps=num_train_steps,
        num_cycles=CFG.num_cycles,
    )

    criterion = HybridLoss()
    awp = AWP(
        model,
        optimizer,
        adv_lr=CFG.awp_lr,
        adv_eps=CFG.awp_eps,
        start_epoch=CFG.awp_start_epoch,
    )

    # 7. Training Loop
    best_score = -np.inf
    final_val_preds = None

    for epoch in range(CFG.epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{CFG.epochs}")

        # Train
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device, awp
        )

        # Validate
        val_preds = valid_fn(val_loader, model, CFG.device)

        # Compute Metric
        score, overall_auc, sub_auc, bpsn_auc, bnsp_auc = compute_bias_metrics(
            val_df, val_preds
        )

        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Score: {score:.6f}")
        logger.info(f"  Overall AUC: {overall_auc:.6f}")

        if score > best_score:
            best_score = score
            final_val_preds = val_preds
            # We don't save the model to disk here to save time/space in this fast baseline,
            # but in a real run we would. We keep the current model in memory.

    # 8. Final Evaluation & Analysis
    logger.info("\n" + "=" * 30)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {best_score}")
    logger.info("=" * 30)

    if final_val_preds is not None:
        failure_analysis(val_df, final_val_preds)

    # 9. Submission Logic
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.9268315106992828

    if best_score > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation score ({best_score}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_loader = get_test_loader(test_df, tokenizer)

        # Inference
        test_preds = inference_fn(test_loader, model, CFG.device)

        # Save
        submission = pd.DataFrame({"id": test_df["id"], "prediction": test_preds})
        submission.to_csv(CFG.submission_path, index=False)
        logger.info(f"Submission saved to {CFG.submission_path}")
    else:
        logger.info(
            f"Validation score ({best_score}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
