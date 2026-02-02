import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from transformers import AutoTokenizer

# Import library modules
from library.config import Config
from library.utils import set_seed, JigsawMetric
from library.data import load_data, get_weighted_loader, DataMiner, ToxicityDataset
from library.model import ToxicityModel, AWP
from library.losses import JigsawLoss
from library.engine import train_fn, eval_fn, inference_fn, run_mlm


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    set_seed(Config.seed)
    device = Config.device
    print(f"Device: {device}")

    # Override Config for Fast Baseline Execution
    # A100 can handle larger batches, speeding up execution
    Config.train_batch_size = 16
    Config.valid_batch_size = 64  # Increased for faster inference
    Config.epochs = 1
    Config.dapt_epochs = 1
    Config.scout_epochs = 1

    # Subsampling limits for speed
    TRAIN_LIMIT = 1500
    VAL_LIMIT = 500
    MINING_POOL_LIMIT = 1000

    # ==========================================
    # 2. Data Loading & Preprocessing
    # ==========================================
    print("\nLoading and Subsampling Data...")
    train_df = load_data("train")
    val_df = load_data("val")
    test_df = load_data("test")

    # Subsample for speed
    train_df = train_df.iloc[:TRAIN_LIMIT].reset_index(drop=True)
    val_df = val_df.iloc[:VAL_LIMIT].reset_index(drop=True)
    # We keep a subset of test for mining, but need full test for final submission if threshold met
    mining_pool_df = test_df.iloc[:MINING_POOL_LIMIT].reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Mining Pool size: {len(mining_pool_df)}")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # ==========================================
    # 3. Stage 1: Domain-Adaptive Pretraining (DAPT)
    # ==========================================
    print("\n=== Stage 1: Domain-Adaptive Pretraining ===")
    dapt_texts = train_df[Config.text_col].tolist()
    # Run MLM on the small subset
    dapt_model_path = run_mlm(dapt_texts, [], tokenizer, device)

    # ==========================================
    # 4. Stage 2: Scout Training & Mining
    # ==========================================
    print("\n=== Stage 2: Scout Training & Mining ===")
    # Initialize Scout Model (using DAPT weights)
    # We use the same architecture but potentially fewer epochs or smaller model in full solution.
    # Here we use the DAPT model path.
    scout_model = ToxicityModel(model_name=dapt_model_path, pretrained=True)
    scout_model.to(device)

    scout_optimizer = AdamW(scout_model.parameters(), lr=Config.scout_lr)
    scout_loss_fn = JigsawLoss()

    # Train Scout
    train_loader = get_weighted_loader(train_df, tokenizer, Config.train_batch_size)
    train_fn(
        scout_model, train_loader, scout_optimizer, None, scout_loss_fn, device, epoch=0
    )

    # Mine Hard Negatives
    print("Mining Hard Negatives...")
    mining_loader = get_weighted_loader(
        mining_pool_df, tokenizer, Config.valid_batch_size, is_test=True
    )
    scout_preds = inference_fn(scout_model, mining_loader, device)

    # Prepare data for miner
    scout_preds_df = mining_pool_df.copy()
    scout_preds_df["prediction"] = scout_preds["toxicity"]

    miner = DataMiner()
    augmented_train_df = miner.augment_training_data(
        train_df, scout_preds_df, scout_preds["identity"]
    )

    # Clean up Scout to free memory
    del scout_model, scout_optimizer, train_loader, mining_loader, scout_preds
    torch.cuda.empty_cache()

    # ==========================================
    # 5. Stage 3: Final Robust Training
    # ==========================================
    print("\n=== Stage 3: Final Robust Training ===")
    final_model = ToxicityModel(model_name=dapt_model_path, pretrained=True)
    final_model.to(device)

    optimizer = AdamW(
        final_model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler
    num_training_steps = (
        len(augmented_train_df) // Config.train_batch_size
    ) * Config.epochs
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # AWP
    awp = AWP(
        final_model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )

    # Loss
    loss_fn = JigsawLoss()

    # Data Loader for Augmented Data
    aug_train_loader = get_weighted_loader(
        augmented_train_df, tokenizer, Config.train_batch_size
    )

    # Training Loop
    for epoch in range(Config.epochs):
        print(f"Epoch {epoch + 1}/{Config.epochs}")
        train_fn(
            final_model,
            aug_train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            awp=awp,
            epoch=epoch,
        )

    # ==========================================
    # 6. Validation & Failure Analysis
    # ==========================================
    print("\n=== Validation ===")
    # Manually create validation loader to ensure deterministic evaluation (no weighted sampling)
    # We need targets, so is_test=False, but we want shuffle=False and no sampler.
    val_dataset = ToxicityDataset(val_df, tokenizer, Config.max_len, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    losses, metrics, val_preds = eval_fn(
        final_model, val_loader, loss_fn, device, val_df
    )

    print(f"Final Validation Metric: {metrics['final_score']}")

    print("\nFailure Analysis (Correlation of Absolute Error with Identity):")
    val_df["abs_error"] = np.abs(val_df[Config.binary_target_col] - val_preds)
    for col in Config.identity_cols:
        if col in val_df.columns:
            # Fill NaNs with 0 for correlation check
            corr = val_df[col].fillna(0.0).corr(val_df["abs_error"])
            print(f"  {col}: {corr:.4f}")

    # ==========================================
    # 7. Submission
    # ==========================================
    threshold = 0.9268315106992828
    if metrics["final_score"] > threshold:
        print(
            f"\nMetric {metrics['final_score']} > {threshold}. Generating Submission..."
        )

        # Load FULL test set for submission
        full_test_df = load_data("test", load_cached_data=True)

        # Create loader
        test_loader = get_weighted_loader(
            full_test_df, tokenizer, Config.valid_batch_size, is_test=True
        )

        # Inference
        preds = inference_fn(final_model, test_loader, device)

        # Create submission file
        submission = pd.DataFrame(
            {"id": full_test_df["id"], "prediction": preds["toxicity"]}
        )

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nMetric {metrics['final_score']} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
