import os
import torch
import pandas as pd
import numpy as np
import gc
import time
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from scipy.stats import pearsonr

# Import library modules
from library.config import CFG
from library.utils import seed_everything, JigsawMetrics, AWP, EMA
from library.losses import JigsawLoss
from library.data import get_loaders, get_test_loader, preprocess_data
from library.model import JigsawModel
from library.engine import train_mlm, train_epoch, valid_epoch, inference


# Wrapper to limit training steps for fast baseline execution
class LimitLoader:
    def __init__(self, loader, limit_steps):
        self.loader = loader
        self.limit_steps = limit_steps
        self.batch_size = loader.batch_size

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.limit_steps:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.limit_steps)


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    device = CFG.device

    # Configuration for Fast Baseline
    # Limit steps per epoch to ensure completion within 2 hours
    STEPS_PER_EPOCH = 500
    VAL_STEPS_DURING_TRAIN = 200

    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # ==========================================
    # Stage 1: Domain-Adaptive Pretraining (DAPT)
    # ==========================================
    print("\n=== Stage 1: Domain-Adaptive Pretraining (DAPT) ===")

    # Load MLM Data
    dapt_loader, _ = get_loaders("dapt", tokenizer, load_cached_data=True)
    dapt_loader = LimitLoader(dapt_loader, STEPS_PER_EPOCH)

    # Initialize MLM Model
    mlm_model = AutoModelForMaskedLM.from_pretrained(CFG.model_name)
    mlm_model.to(device)

    # Optimizer for DAPT
    dapt_optimizer = AdamW(
        mlm_model.parameters(), lr=CFG.dapt_lr, weight_decay=CFG.weight_decay
    )

    # Train DAPT
    for epoch in range(CFG.dapt_epochs):
        avg_loss = train_mlm(
            mlm_model, dapt_loader, dapt_optimizer, None, device, epoch
        )
        print(f"DAPT Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Save backbone weights to transfer
    # DebertaV2ForMaskedLM stores the base model in .deberta
    backbone_state_dict = mlm_model.deberta.state_dict()

    # Cleanup
    del mlm_model, dapt_loader, dapt_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # Stage 2: General Multi-Task Fine-Tuning
    # ==========================================
    print("\n=== Stage 2: General Multi-Task Fine-Tuning ===")

    # Load Classification Data (General)
    train_loader_gen, val_loader = get_loaders(
        "general", tokenizer, load_cached_data=True
    )
    train_loader_gen = LimitLoader(train_loader_gen, STEPS_PER_EPOCH)
    # Create a small validation loader for monitoring during training
    val_loader_small = LimitLoader(val_loader, VAL_STEPS_DURING_TRAIN)

    # Initialize Jigsaw Model
    model = JigsawModel(pretrained=True)
    # Load DAPT weights into backbone
    model.backbone.load_state_dict(backbone_state_dict)
    model.to(device)

    # Optimizer & Scheduler
    # Separate LR for encoder and heads
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": CFG.encoder_lr},
        {"params": model.toxicity_head.parameters(), "lr": CFG.decoder_lr},
        {"params": model.identity_head.parameters(), "lr": CFG.decoder_lr},
        {"params": model.attack_head.parameters(), "lr": CFG.decoder_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=CFG.weight_decay)

    num_train_steps = STEPS_PER_EPOCH * CFG.stage2_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * CFG.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    criterion = JigsawLoss()

    # Train Stage 2
    for epoch in range(CFG.stage2_epochs):
        train_epoch(
            model,
            train_loader_gen,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            stage="general",
        )
        # Validate on subset
        valid_epoch(model, val_loader_small, criterion, device)

    # Cleanup Stage 2 loader
    del train_loader_gen
    torch.cuda.empty_cache()

    # ==========================================
    # Stage 3: Robust Metric Optimization
    # ==========================================
    print("\n=== Stage 3: Robust Metric Optimization ===")

    # Load Classification Data (Robust)
    train_loader_rob, _ = get_loaders("robust", tokenizer, load_cached_data=True)
    train_loader_rob = LimitLoader(train_loader_rob, STEPS_PER_EPOCH)

    # Re-init Optimizer with lower LR for refinement
    optimizer = AdamW(
        model.parameters(), lr=CFG.stage3_lr, weight_decay=CFG.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=STEPS_PER_EPOCH * CFG.stage3_epochs,
    )

    # Initialize AWP and EMA
    awp = (
        AWP(model, optimizer, adv_lr=CFG.awp_lr, adv_eps=CFG.awp_eps)
        if CFG.use_awp
        else None
    )
    ema = EMA(model, decay=CFG.ema_decay) if CFG.use_ema else None

    # Train Stage 3
    for epoch in range(CFG.stage3_epochs):
        train_epoch(
            model,
            train_loader_rob,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            stage="robust",
            awp=awp,
            ema=ema,
        )
        # Validate on subset with EMA
        valid_epoch(model, val_loader_small, criterion, device, ema=ema)

    # ==========================================
    # Final Evaluation & Failure Analysis
    # ==========================================
    print("\n=== Final Evaluation ===")

    # Calculate final metric on FULL validation set
    # valid_epoch returns (avg_loss, final_score)
    _, final_score = valid_epoch(model, val_loader, criterion, device, ema=ema)

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")

    # Get predictions and targets for analysis
    if ema:
        ema.apply_shadow()
    model.eval()

    val_preds = []
    val_targets = []

    # We need to load the validation dataframe to get metadata features
    # preprocess_data caches the dataframes, so this is fast
    _, val_df, _ = preprocess_data(load_cached_data=True)

    # Run inference on val set manually to get raw probabilities for analysis
    # We iterate over val_loader (which is not shuffled)
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
            val_preds.append(preds)
            val_targets.append(batch["target"].numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()

    if ema:
        ema.restore()

    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_preds)

    # Features to correlate
    # 1. Text Length
    texts = val_df["comment_text"].fillna("").astype(str).values
    text_lens = np.array([len(t) for t in texts])

    # 2. Identity Columns
    identity_cols = CFG.identity_cols

    print("Correlation between Error Magnitude and Features:")

    # Length Correlation
    corr_len, _ = pearsonr(errors, text_lens)
    print(f"  Text Length: {corr_len:.4f}")

    # Identity Correlations
    for col in identity_cols:
        if col in val_df.columns:
            # Handle NaNs in identities (assume 0 for correlation)
            col_vals = val_df[col].fillna(0.0).values
            corr_id, _ = pearsonr(errors, col_vals)
            print(f"  Identity '{col}': {corr_id:.4f}")

    # ==========================================
    # Submission
    # ==========================================
    THRESHOLD = 0.9268315106992828

    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader, test_ids = get_test_loader(tokenizer, load_cached_data=True)

        # Use EMA weights for final inference
        predictions = inference(model, test_loader, device, ema=ema)

        submission = pd.DataFrame({"id": test_ids, "prediction": predictions.flatten()})

        sub_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
