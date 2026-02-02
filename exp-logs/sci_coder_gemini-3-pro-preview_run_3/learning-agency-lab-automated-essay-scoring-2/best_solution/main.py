import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_llrd_optimizer_params, compute_qwk
from library.data import get_dataloaders, process_data
from library.model import EssayModel
from library.engine import run_training
from library.stacking import extract_features, train_lgbm, predict_stacking


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    # Set seeds for reproducibility
    seed_everything(Config.seed)

    # Enable TF32 for faster training on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Optimize Config for Fast Baseline execution
    # Reducing epochs to ensure completion within time limits while maintaining performance
    Config.epochs = 2

    print(f"Device: {Config.device}")
    print(f"Model: {Config.model_name}")
    print(f"Epochs: {Config.epochs}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("\n=== Loading Data ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load dataloaders (uses cached processed data if available)
    train_loader, val_loader = get_dataloaders(tokenizer, load_cached_data=True)

    # =========================================================================
    # 3. Stage 1: Backbone Fine-Tuning
    # =========================================================================
    print("\n=== Stage 1: Backbone Fine-Tuning ===")

    # Initialize Model
    model = EssayModel(pretrained=True)
    model.to(Config.device)

    # Setup Optimizer with Layer-wise Learning Rate Decay (LLRD)
    optimizer_params = get_llrd_optimizer_params(
        model,
        encoder_lr=Config.backbone_lr,
        head_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )
    optimizer = AdamW(optimizer_params, lr=Config.head_lr, eps=1e-6)

    # Setup Scheduler
    num_training_steps = (
        len(train_loader) * Config.epochs // Config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * Config.num_warmup_steps_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Run Training Loop
    best_qwk_stage1 = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        Config.device,
        Config.epochs,
        patience=2,
    )

    # Load best model weights for Stage 2
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    # =========================================================================
    # 4. Stage 2: Stacking (Feature Extraction & LightGBM)
    # =========================================================================
    print("\n=== Stage 2: Stacking ===")

    # Load processed dataframes to access text and meta-features
    train_df = process_data(Config.TRAIN_META, "train", load_cached_data=True)
    val_df = process_data(Config.VAL_META, "val", load_cached_data=True)

    # Extract Features (Embeddings + Meta-features)
    # This handles sliding window inference and pooling
    train_feats, train_labels, _ = extract_features(
        train_df, model, tokenizer, Config.device, "train"
    )
    val_feats, val_labels, _ = extract_features(
        val_df, model, tokenizer, Config.device, "val"
    )

    # Train LightGBM Head
    lgbm_model = train_lgbm(train_feats, train_labels, val_feats, val_labels)

    # =========================================================================
    # 5. Validation Assessment
    # =========================================================================
    print("\n=== Validation Assessment ===")

    # Generate predictions on validation set
    val_preds = lgbm_model.predict(val_feats)

    # Compute QWK
    # Note: compute_qwk handles clipping and rounding internally, but we do it here
    # to ensure consistency with the metric variable we print.
    val_preds_rounded = np.clip(val_preds, 1, 6).round().astype(int)
    val_labels_int = val_labels.astype(int)

    final_metric = compute_qwk(val_labels_int, val_preds_rounded)

    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    errors = np.abs(val_labels - val_preds)

    # Features to analyze for correlation with error
    analysis_features = [
        "word_count",
        "sentence_count",
        "unique_word_ratio",
        "avg_word_len",
    ]

    print("Correlation between Error Magnitude and Input Features:")
    for feat in analysis_features:
        if feat in val_df.columns:
            feat_values = val_df[feat].values
            # Calculate Pearson correlation
            if len(feat_values) == len(errors):
                corr = np.corrcoef(feat_values, errors)[0, 1]
                print(f"  {feat}: {corr:.6f}")
            else:
                print(f"  {feat}: Dimension mismatch")
        else:
            print(f"  {feat}: Not found in metadata")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    print("\n=== Submission Generation ===")

    THRESHOLD = 0.8174385126572309

    if final_metric > THRESHOLD:
        print(
            f"Validation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_df = process_data(Config.TEST_META, "test", load_cached_data=True)

        # Extract Test Features
        test_feats, _, test_ids = extract_features(
            test_df, model, tokenizer, Config.device, "test"
        )

        # Predict and Save
        predict_stacking(lgbm_model, test_feats, test_ids)

    else:
        print(
            f"Validation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
