import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_mlm_loaders, prepare_kfold_loaders, prepare_test_loader
from library.model import CustomDeberta
from library.awp import AWP
from library.engine import train_mlm, train_fn, valid_fn, inference_fn

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    seed_everything(Config.seed)

    # Override Config for Fast Baseline Execution
    # We aim to complete within ~1.5 hours using the A100 GPU
    Config.mlm_epochs = 1  # Reduced from 3 to 1
    Config.epochs = 3  # Reduced from 5 to 3
    Config.n_folds = 1  # Run only Fold 0
    Config.num_workers = 8  # Utilize available vCPUs

    print("=" * 40)
    print("RUN CONFIGURATION")
    print("=" * 40)
    print(f"Device: {Config.device}")
    print(f"MLM Epochs: {Config.mlm_epochs}")
    print(f"Supervised Epochs: {Config.epochs}")
    print(f"Folds to Train: {Config.n_folds}")
    print(f"Use AWP: {Config.use_awp}")
    print("-" * 40)

    # ==========================================
    # 2. Stage 1: Domain-Adaptive Pre-training
    # ==========================================
    print("\n" + "=" * 40)
    print("STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING (MLM)")
    print("=" * 40)

    # Prepare MLM Data
    mlm_loader = prepare_mlm_loaders(load_cached_data=True)

    # Train MLM Backbone
    train_mlm(mlm_loader, Config.device, epochs=Config.mlm_epochs)

    # ==========================================
    # 3. Stage 2: Supervised Fine-Tuning
    # ==========================================
    print("\n" + "=" * 40)
    print("STAGE 2: SUPERVISED FINE-TUNING")
    print("=" * 40)

    # We only train Fold 0 for this baseline run
    fold = 0
    print(f"Starting training for Fold {fold}...")

    # Prepare Data
    train_loader, val_loader = prepare_kfold_loaders(fold, load_cached_data=True)

    # Initialize Model with DAPT weights
    # The checkpoint path is where train_mlm saved the model
    model = CustomDeberta(pretrained=True, checkpoint_path=Config.mlm_model_dir)
    model.to(Config.device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.pct_start,
        anneal_strategy="cos",
    )

    # Adversarial Weight Perturbation (AWP)
    awp = None
    if Config.use_awp:
        print("Initializing AWP...")
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

    # Training Loop
    best_score = -1
    best_model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.pth")

    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")

        # Train Step
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, Config.device, epoch, awp
        )
        print(f"  Train Loss: {train_loss:.5f}")

        # Validation Step
        val_loss, val_score = valid_fn(val_loader, model, Config.device)
        print(f"  Val Loss: {val_loss:.5f} | Val AUC: {val_score:.5f}")

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best Score! Model saved to {best_model_path}")

    # Print Final Metric Requirement
    print("\n" + "=" * 40)
    print(f"Final Validation Metric: {best_score}")
    print("=" * 40)

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
    model.eval()

    # Collect predictions and targets
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]

            val_preds.append(torch.sigmoid(logits).cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Mean Absolute Error per sample
    errors = np.abs(val_targets - val_preds).mean(axis=1)

    # Get text lengths from the dataset
    val_texts = val_loader.dataset.texts
    lengths = np.array([len(str(t).split()) for t in val_texts])

    # Calculate Correlation
    if len(errors) == len(lengths):
        corr = np.corrcoef(lengths, errors)[0, 1]
        print(f"Correlation between Error Magnitude and Word Count: {corr:.6f}")
    else:
        print("Error: Length mismatch between predictions and texts.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9920879090652149

    if best_score > THRESHOLD:
        print(
            f"\nScore {best_score} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Prepare Test Loader
        test_loader, test_ids = prepare_test_loader(load_cached_data=True)

        # Inference
        test_preds = inference_fn(test_loader, model, Config.device)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_preds, columns=Config.target_cols)
        submission_df.insert(0, "id", test_ids)

        # Save
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nScore {best_score} does not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
