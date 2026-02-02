import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger, compute_score
from library.cpc_loader import ContextMapper
from library.dataset import PhraseDataset
from library.model import ScalarMixingModel
from library.loss import HybridLoss
from library.awp import AWP
from library.ema import ModelEMA
from library.engine import train_fn, valid_fn, inference_fn


def run():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    seed_everything(Config.seed)
    logger = get_logger("runfile.log")

    # Override Config for Fast Baseline Execution
    Config.epochs = 2
    Config.awp_start_epoch = 1  # Apply AWP in the second epoch (index 1)
    Config.submission_path = "./submission/submission.csv"

    # Create submission directory
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    logger.info(
        f"Configuration Configured: Epochs={Config.epochs}, Device={Config.device}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    logger.info("Loading Metadata...")
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # Context Mapping
    logger.info("Mapping Contexts...")
    mapper = ContextMapper()
    # Fit on all available contexts to ensure coverage
    all_contexts = pd.concat(
        [df_train["context"], df_val["context"], df_test["context"]]
    ).unique()
    mapper.fit(all_contexts)

    # Apply mapping
    df_train["context_text"] = df_train["context"].map(mapper.context_map).fillna("")
    df_val["context_text"] = df_val["context"].map(mapper.context_map).fillna("")
    df_test["context_text"] = df_test["context"].map(mapper.context_map).fillna("")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Datasets
    # We use the provided stratified split (train.csv / val.csv) directly
    train_dataset = PhraseDataset(df_train, tokenizer, max_length=Config.max_length)
    val_dataset = PhraseDataset(df_val, tokenizer, max_length=Config.max_length)
    test_dataset = PhraseDataset(
        df_test, tokenizer, max_length=Config.max_length, is_test=True
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.inference_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model, Optimizer (LLRD), Scheduler
    # -------------------------------------------------------------------------
    logger.info("Initializing Model...")
    model = ScalarMixingModel(pretrained=True)
    model.to(Config.device)

    # Layer-wise Learning Rate Decay (LLRD) Setup
    optimizer_parameters = []
    lr = Config.learning_rate
    decay = Config.llrd_decay

    # Group parameters by layer
    # DeBERTa V3 Large structure: backbone.embeddings, backbone.encoder.layer.{0..23}, head
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" not in name:
            layer_id = 25  # Task Head / Pooling
        elif "embeddings" in name:
            layer_id = 0
        elif "encoder.layer" in name:
            try:
                # Extract layer index: backbone.encoder.layer.15.output... -> 15
                layer_id = int(name.split("encoder.layer.")[1].split(".")[0]) + 1
            except:
                layer_id = 0
        else:
            layer_id = 0

        if layer_id not in groups:
            groups[layer_id] = []
        groups[layer_id].append(param)

    # Assign LRs
    for layer_id, params in groups.items():
        if layer_id == 25:
            cur_lr = lr * Config.head_lr_scale
        else:
            # Decay from top to bottom
            cur_lr = lr * (decay ** (25 - layer_id))

        optimizer_parameters.append(
            {"params": params, "lr": cur_lr, "weight_decay": Config.weight_decay}
        )

    optimizer = AdamW(optimizer_parameters, eps=Config.adam_epsilon)

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # AWP & EMA
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )
    ema = ModelEMA(model, decay=Config.ema_decay, device=Config.device)
    loss_fn = HybridLoss()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting Training...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            Config.device,
            scheduler,
            epoch,
            Config,
            awp,
            ema,
            loss_fn,
        )

        # Validate (using EMA weights temporarily)
        val_loss, val_score = valid_fn(
            val_loader, model, Config.device, Config, ema, loss_fn
        )

        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pearson: {val_score:.4f}"
        )

    # -------------------------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Running Final Evaluation...")

    # Apply EMA weights permanently for inference
    ema.apply_shadow()

    # Predict on Validation Set
    val_preds = []
    val_targets = []
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            token_type_ids = batch["token_type_ids"].to(Config.device)
            labels = batch["label"].to(Config.device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            # Regression logits are the first output
            val_preds.append(outputs[0].view(-1).cpu().numpy())
            val_targets.append(labels.view(-1).cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_preds = np.clip(val_preds, 0, 1)

    final_score = compute_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(val_preds - val_targets)

    # Calculate feature correlations with error
    # Note: df_val order matches val_loader (shuffle=False)
    anchor_lens = df_val["anchor"].astype(str).apply(len).values
    target_lens = df_val["target"].astype(str).apply(len).values

    corr_anchor, _ = pearsonr(errors, anchor_lens)
    corr_target, _ = pearsonr(errors, target_lens)

    print(f"Error Correlation with Anchor Length: {corr_anchor:.6f}")
    print(f"Error Correlation with Target Length: {corr_target:.6f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    threshold = 0.8698034882545471
    if final_score > threshold:
        logger.info(
            f"Score ({final_score:.4f}) > Threshold ({threshold:.4f}). Generating Submission..."
        )

        test_preds = inference_fn(
            test_loader, model, Config.device, Config, ema=None
        )  # EMA already applied

        submission = pd.DataFrame({"id": df_test["id"], "score": test_preds})

        submission.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")
    else:
        logger.info(
            f"Score ({final_score:.4f}) did not meet threshold. Submission skipped."
        )


if __name__ == "__main__":
    run()
