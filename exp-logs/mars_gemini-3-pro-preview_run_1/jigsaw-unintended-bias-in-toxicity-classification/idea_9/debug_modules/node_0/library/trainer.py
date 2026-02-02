import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score

from library.config import CFG
from library.utils import get_logger, AverageMeter, timeSince, seed_everything
from library.data_processing import (
    get_data,
    get_loaders,
    get_mlm_loader,
    get_test_loader,
    get_tokenizer,
)
from library.model import JigsawModel
from library.losses import HybridLoss
from library.awp import AWP

# Initialize Logger
logger = get_logger()


# ====================================================
# Metric Calculation
# ====================================================
def calculate_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def compute_bias_metrics(df, preds):
    """
    Computes the competition metric:
    Score = 0.25*Overall_AUC + 0.25*Mean(Subgroup_AUC) + 0.25*Mean(BPSN_AUC) + 0.25*Mean(BNSP_AUC)
    """
    y_true = (df["target"].values >= 0.5).astype(int)
    y_pred = preds

    # Overall AUC
    overall_auc = calculate_auc(y_true, y_pred)

    # Bias AUCs
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []

    for col in CFG.identity_cols:
        # Identity mask (assuming binary mention if >= 0.5, consistent with task)
        # Note: Metadata contains fractional values, but task implies mentions are specific subsets.
        # Standard approach is boolean mask for evaluation.
        ident_mask = df[col].values >= 0.5

        # Skip if identity not present
        if ident_mask.sum() == 0:
            continue

        # 1. Subgroup AUC
        # Restrict to examples mentioning the identity
        sub_auc = calculate_auc(y_true[ident_mask], y_pred[ident_mask])
        subgroup_aucs.append(sub_auc)

        # 2. BPSN AUC (Background Positive, Subgroup Negative)
        # Non-toxic examples mentioning identity (Subgroup Negative)
        # AND Toxic examples NOT mentioning identity (Background Positive)
        sub_neg = ident_mask & (y_true == 0)
        back_pos = (~ident_mask) & (y_true == 1)
        bpsn_mask = sub_neg | back_pos
        bpsn_auc = calculate_auc(y_true[bpsn_mask], y_pred[bpsn_mask])
        bpsn_aucs.append(bpsn_auc)

        # 3. BNSP AUC (Background Negative, Subgroup Positive)
        # Toxic examples mentioning identity (Subgroup Positive)
        # AND Non-toxic examples NOT mentioning identity (Background Negative)
        sub_pos = ident_mask & (y_true == 1)
        back_neg = (~ident_mask) & (y_true == 0)
        bnsp_mask = sub_pos | back_neg
        bnsp_auc = calculate_auc(y_true[bnsp_mask], y_pred[bnsp_mask])
        bnsp_aucs.append(bnsp_auc)

    # Generalized Mean
    def power_mean(metrics, p):
        if not metrics:
            return 0.5
        total = sum([m**p for m in metrics])
        return (total / len(metrics)) ** (1 / p)

    # Calculate components
    # If lists are empty (unlikely given dataset), default to 0.5
    avg_subgroup = power_mean(subgroup_aucs, CFG.metric_p) if subgroup_aucs else 0.5
    avg_bpsn = power_mean(bpsn_aucs, CFG.metric_p) if bpsn_aucs else 0.5
    avg_bnsp = power_mean(bnsp_aucs, CFG.metric_p) if bnsp_aucs else 0.5

    final_score = (
        (0.25 * overall_auc)
        + (0.25 * avg_subgroup)
        + (0.25 * avg_bpsn)
        + (0.25 * avg_bnsp)
    )

    return final_score, overall_auc, avg_subgroup, avg_bpsn, avg_bnsp


# ====================================================
# Domain Adaptive Pretraining (DAPT)
# ====================================================
def run_dapt(train_df, test_df, tokenizer):
    """
    Runs Masked Language Modeling on combined Train + Test data.
    Saves the backbone weights to cache.
    """
    logger.info("Starting Domain-Adaptive Pretraining (MLM)...")

    # Prepare DataLoader
    train_loader = get_mlm_loader(train_df, test_df, tokenizer)

    # Initialize Model for MLM
    model = AutoModelForMaskedLM.from_pretrained(CFG.model_name)
    model.to(CFG.device)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CFG.mlm_lr, weight_decay=CFG.weight_decay)

    # Training Loop
    # We only run for defined epochs (usually 1)
    for epoch in range(CFG.mlm_epochs):
        start_time = time.time()
        losses = AverageMeter()

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(CFG.device) for k, v in batch.items()}

            # Transformers handles masking internally via DataCollator usually,
            # but since we didn't use a collator in get_mlm_loader, we rely on
            # the model's forward pass to calculate loss if labels are provided.
            # However, AutoModelForMaskedLM expects 'labels'.
            # We must create labels and mask input_ids manually if not using a collator.
            # Simplified approach: Use a basic masking strategy here.

            input_ids = inputs["input_ids"].clone()
            labels = inputs["input_ids"].clone()

            # Create mask
            probability_matrix = torch.full(labels.shape, CFG.mlm_mask_prob).to(
                CFG.device
            )
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, replace masked input tokens with tokenizer.mask_token_id
            indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(CFG.device)
                & masked_indices
            )
            input_ids[indices_replaced] = tokenizer.mask_token_id

            # 10% of the time, replace masked input tokens with random word
            # (Skipping for brevity/speed, keeping 80% mask, 20% original/random mix implicitly by not changing others)

            outputs = model(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss.item(), input_ids.size(0))

            if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
                logger.info(f"MLM Epoch {epoch+1} Step {step} Loss {losses.avg:.4f}")

        logger.info(f"MLM Epoch {epoch+1} finished. Time: {timeSince(start_time, 1.0)}")

    # Save Backbone Weights
    # DeBERTa-v3 structure: model.deberta
    backbone_path = os.path.join(CFG.cache_dir, "dapt_backbone.pth")
    # We save the base model state dict (e.g. 'deberta') to be compatible with JigsawModel
    # JigsawModel uses AutoModel.from_pretrained, which loads the base model.
    # We extract the base model from AutoModelForMaskedLM.
    if hasattr(model, "deberta"):
        torch.save(model.deberta.state_dict(), backbone_path)
    elif hasattr(model, "roberta"):
        torch.save(model.roberta.state_dict(), backbone_path)
    elif hasattr(model, "bert"):
        torch.save(model.bert.state_dict(), backbone_path)
    else:
        # Fallback: try saving base_model
        torch.save(model.base_model.state_dict(), backbone_path)

    logger.info(f"DAPT Backbone saved to {backbone_path}")

    del model, optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    return backbone_path


# ====================================================
# Supervised Training Helper Functions
# ====================================================
def train_fn(
    train_loader, model, criterion, optimizer, epoch, scheduler, device, awp=None
):
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        aux_labels = batch["aux_labels"].to(device)
        loss_weights = batch["loss_weight"].to(device)

        batch_size = input_ids.size(0)

        # Forward Pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets, aux_labels, loss_weights)

        # Backward
        loss.backward()

        # Adversarial Weight Perturbation
        if CFG.use_awp and epoch >= CFG.awp_start_epoch and awp is not None:
            awp.attack()
            # Re-forward with perturbed weights
            outputs_adv = model(input_ids, attention_mask)
            loss_adv = criterion(outputs_adv, targets, aux_labels, loss_weights)
            loss_adv.backward()
            awp.restore()

        # Gradient Clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        # Optimizer Step
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            logger.info(
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step+1)/len(train_loader))} "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Grad: {grad_norm:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses.avg


def valid_fn(val_loader, model, device):
    model.eval()
    preds = []
    start = time.time()

    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # We only use the main toxicity logits for prediction
            logits = outputs["logits"]

        # Sigmoid to get probabilities
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        preds.append(probs)

    predictions = np.concatenate(preds)
    return predictions


def inference_fn(test_loader, model, device):
    model.eval()
    preds = []

    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]

        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        preds.append(probs)

    predictions = np.concatenate(preds)
    return predictions


# ====================================================
# Main Training Loop
# ====================================================
def train_loop():
    seed_everything(CFG.seed)

    # 1. Data Loading
    tokenizer = get_tokenizer()
    train_df, val_df, test_df = get_data(load_cached_data=True)

    # 2. Domain Adaptation (Optional but recommended in solution)
    # Check if backbone already exists to skip re-training if restarting
    backbone_path = os.path.join(CFG.cache_dir, "dapt_backbone.pth")
    if not os.path.exists(backbone_path):
        run_dapt(train_df, test_df, tokenizer)
    else:
        logger.info("Found cached DAPT backbone. Skipping MLM.")

    # 3. Prepare Loaders
    train_loader, val_loader = get_loaders(train_df, val_df, tokenizer)

    # 4. Model Initialization
    model = JigsawModel(pretrained=True)

    # Load DAPT weights
    if os.path.exists(backbone_path):
        logger.info(f"Loading DAPT weights from {backbone_path}")
        state_dict = torch.load(backbone_path, map_location="cpu")
        # Load into the backbone (model.model)
        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"DAPT Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    model.to(CFG.device)

    # 5. Optimization Setup
    # Differential Learning Rates
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

    # 6. Training Loop
    best_score = -np.inf
    best_model_path = os.path.join(CFG.output_dir, "best_model.bin")

    for epoch in range(CFG.epochs):
        start_time = time.time()

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

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s"
        )
        logger.info(f"Epoch {epoch+1} - Score: {score:.6f}")
        logger.info(f"Epoch {epoch+1} - Overall AUC: {overall_auc:.6f}")
        logger.info(f"Epoch {epoch+1} - Subgroup AUC: {sub_auc:.6f}")
        logger.info(f"Epoch {epoch+1} - BPSN AUC: {bpsn_auc:.6f}")
        logger.info(f"Epoch {epoch+1} - BNSP AUC: {bnsp_auc:.6f}")

        # Save Best
        if score > best_score:
            best_score = score
            logger.info(
                f"Epoch {epoch+1} - Best Score Updated: {best_score:.6f}. Saving model..."
            )
            torch.save(model.state_dict(), best_model_path)

    # 7. Final Inference
    logger.info("Training complete. Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=CFG.device))

    test_loader = get_test_loader(test_df, tokenizer)
    test_preds = inference_fn(test_loader, model, CFG.device)

    # 8. Submission
    logger.info(f"Generating submission file at {CFG.submission_path}...")
    submission = pd.DataFrame({"id": test_df["id"], "prediction": test_preds})
    submission.to_csv(CFG.submission_path, index=False)
    logger.info("Submission saved successfully.")


if __name__ == "__main__":
    train_loop()
