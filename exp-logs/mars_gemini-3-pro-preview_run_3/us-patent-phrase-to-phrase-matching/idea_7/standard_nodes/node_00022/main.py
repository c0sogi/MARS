import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from library
from library.config import Config
from library.dataset import load_processed_data, PearsonDataset
from library.model import CustomDeberta
from library.loss import CompositeLoss
from library.awp import AWP
from library.ema import ModelEMA
from library.engine import set_seed, run_dapt, train_fn, eval_fn, predict_fn


def main():
    # 1. Setup & Configuration
    conf = Config()

    # Adjust configuration for Fast Baseline execution
    conf.epochs = 2
    conf.dapt_epochs = 1
    conf.train_batch_size = 8
    conf.valid_batch_size = 16

    # Set random seed for reproducibility
    set_seed(conf.seed)

    print(f"Configuration: Epochs={conf.epochs}, DAPT Epochs={conf.dapt_epochs}")

    # 2. Data Loading
    # Load processed datasets (Train, Val, Test) with context expansion
    train_df, val_df, test_df = load_processed_data(load_cached_data=True)

    # Subsample training data for fast baseline (limit to 5000 samples)
    train_subset_size = 5000
    if len(train_df) > train_subset_size:
        print(
            f"Subsampling training data from {len(train_df)} to {train_subset_size} samples."
        )
        train_df = train_df.sample(
            n=train_subset_size, random_state=conf.seed
        ).reset_index(drop=True)

    # 3. Domain-Adaptive Pre-training (DAPT)
    if conf.use_dapt:
        # Run DAPT and save model to conf.dapt_model_path
        run_dapt(conf)
        # Update backbone to point to the adapted model for the downstream task
        conf.model_backbone = conf.dapt_model_path
        print(f"Updated model backbone to: {conf.model_backbone}")

    # 4. Supervised Fine-Tuning Setup
    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(conf.model_backbone)

    # Create Datasets
    train_dataset = PearsonDataset(
        train_df, tokenizer, max_length=conf.max_length, mode="train"
    )
    val_dataset = PearsonDataset(
        val_df, tokenizer, max_length=conf.max_length, mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.train_batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=conf.valid_batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    model = CustomDeberta(model_path=conf.model_backbone, pretrained=True)
    model.to(conf.device)

    # Optimizer (Layer-wise learning rates could be implemented, but using grouped params for simplicity/speed)
    # Group parameters to apply different learning rates to backbone vs head
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters()],
            "lr": conf.encoder_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": conf.decoder_lr,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=conf.encoder_lr,
        eps=conf.eps,
        betas=(conf.beta1, conf.beta2),
        weight_decay=conf.weight_decay,
    )

    # Scheduler
    num_train_steps = int(len(train_df) / conf.train_batch_size * conf.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * conf.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Loss Function
    loss_fn = CompositeLoss(conf)

    # Adversarial Weight Perturbation (AWP)
    awp = None
    if conf.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=conf.awp_lr,
            adv_eps=conf.awp_eps,
            start_epoch=conf.awp_start_epoch,
        )

    # Exponential Moving Average (EMA)
    ema = None
    if conf.use_ema:
        ema = ModelEMA(model, decay=conf.ema_decay)

    # 5. Training Loop
    print("Starting training...")
    best_val_score = -1.0

    for epoch in range(conf.epochs):
        # Train
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            conf.device,
            epoch,
            conf,
            awp,
            ema,
            loss_fn,
        )

        # Validate (using EMA if available)
        val_score = eval_fn(model, val_loader, conf.device, conf, ema)

        print(f"Epoch {epoch+1} Validation Pearson: {val_score:.4f}")

        # Save best model (Raw weights)
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), conf.best_model_path)

    # 6. Final Evaluation & Failure Analysis
    print("\nPerforming Final Evaluation...")

    # We use the final EMA state for the final evaluation if EMA is enabled.
    # Note: eval_fn automatically applies EMA shadow weights if ema is passed.
    final_val_score = eval_fn(model, val_loader, conf.device, conf, ema)

    print(f"Final Validation Metric: {final_val_score}")

    # Failure Analysis
    print("\nRunning Failure Analysis...")
    # Apply EMA weights permanently for analysis/inference if used
    if conf.use_ema and ema is not None:
        ema.apply_shadow(model)

    val_preds = predict_fn(model, val_loader, conf.device, conf)
    val_targets = val_df["score"].values

    # Calculate absolute error
    errors = np.abs(val_preds - val_targets)

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["pred"] = val_preds
    analysis_df["error"] = errors
    analysis_df["anchor_len"] = analysis_df["anchor"].astype(str).apply(len)
    analysis_df["target_len"] = analysis_df["target"].astype(str).apply(len)
    analysis_df["context_len"] = analysis_df["context_text"].astype(str).apply(len)

    print("Correlation between Error Magnitude and Features:")
    for feature in ["anchor_len", "target_len", "context_len", "score"]:
        corr = analysis_df[feature].corr(analysis_df["error"])
        print(f"  {feature}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.8698034882545471
    if final_val_score > threshold:
        print(
            f"\nValidation score ({final_val_score}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Prepare Test Data
        test_dataset = PearsonDataset(
            test_df, tokenizer, max_length=conf.max_length, mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=conf.valid_batch_size,
            shuffle=False,
            num_workers=conf.num_workers,
            pin_memory=True,
        )

        # Predict (Model already has EMA weights applied from Failure Analysis step)
        test_preds = predict_fn(model, test_loader, conf.device, conf)

        # Save Submission
        submission = pd.read_csv(conf.sample_submission_path)
        # Ensure alignment
        if len(submission) != len(test_preds):
            print(
                f"Warning: Submission length {len(submission)} != Preds length {len(test_preds)}"
            )

        submission["score"] = test_preds
        submission.to_csv(conf.submission_path, index=False)
        print(f"Submission saved to {conf.submission_path}")

    else:
        print(
            f"\nValidation score ({final_val_score}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
