import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_optimizer_params, compute_qwk
from library.dataset import EssayDataset, Collate
from library.model import EssayModel
from library.engine import train_loop, generate_submission


def run():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Initialize tokenizer implicitly via Dataset or explicitly if needed.
    # EssayDataset loads AutoTokenizer from Config.model_name by default.

    train_dataset = EssayDataset(
        data_path=Config.train_path,
        processed_path=Config.train_processed_path,
        load_cached_data=True,
        is_test=False,
        debug=Config.debug,
    )

    val_dataset = EssayDataset(
        data_path=Config.val_path,
        processed_path=Config.val_processed_path,
        load_cached_data=True,
        is_test=False,
        debug=Config.debug,
    )

    # Initialize Collator
    collator = Collate(tokenizer=train_dataset.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = EssayModel(
        model_name=Config.model_name, num_labels=Config.num_labels, pretrained=True
    )
    model.to(device)

    # 4. Optimization
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate * 5,  # Higher LR for head
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )

    optimizer = AdamW(optimizer_parameters, lr=Config.learning_rate, eps=1e-6)

    num_train_steps = int(len(train_dataset) / Config.train_batch_size * Config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    print("Starting training...")
    model = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        save_path=Config.model_save_path,
        patience=2,  # Strict patience for fast baseline
    )

    # 6. Validation Assessment & Failure Analysis
    print("Performing validation assessment...")

    # Reload best model to be sure
    if os.path.exists(Config.model_save_path):
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    model.eval()
    val_preds_raw = []
    val_targets_raw = []

    # Manual inference loop to get raw predictions for analysis
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            # Ordinal decoding: 1 + sum(sigmoid(logits))
            probs = torch.sigmoid(logits)
            pred_score = 1.0 + probs.sum(dim=1)
            target_score = 1.0 + labels.sum(dim=1)

            val_preds_raw.extend(pred_score.cpu().numpy())
            val_targets_raw.extend(target_score.cpu().numpy())

    # Process predictions
    val_preds_rounded = np.round(val_preds_raw).astype(int)
    val_preds_clipped = np.clip(val_preds_rounded, 1, 6)
    val_targets_int = np.array(val_targets_raw).astype(int)

    # Compute Metric
    final_qwk = compute_qwk(val_targets_int, val_preds_clipped)
    print(f"Final Validation Metric: {final_qwk}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_targets_int - val_preds_clipped)

    # Get features from validation dataframe
    df_val = val_dataset.df
    # Ensure alignment (dataset order is preserved in loader with shuffle=False)
    # Calculate simple text features
    df_val["char_count"] = df_val["full_text"].astype(str).apply(len)
    df_val["word_count"] = (
        df_val["full_text"].astype(str).apply(lambda x: len(x.split()))
    )

    # Compute correlations
    corr_char = np.corrcoef(errors, df_val["char_count"])[0, 1]
    corr_word = np.corrcoef(errors, df_val["word_count"])[0, 1]

    print(f"Correlation (Error vs Char Count): {corr_char:.4f}")
    print(f"Correlation (Error vs Word Count): {corr_word:.4f}")

    # 7. Submission
    threshold = 0.8203336917523225
    if final_qwk > threshold:
        print(
            f"\nValidation metric {final_qwk} > {threshold}. Generating submission..."
        )

        test_dataset = EssayDataset(
            data_path=Config.test_path,
            processed_path=Config.test_processed_path,
            load_cached_data=True,
            is_test=True,
            debug=Config.debug,
            tokenizer=train_dataset.tokenizer,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        generate_submission(
            model=model,
            test_loader=test_loader,
            device=device,
            output_path=Config.submission_path,
        )
    else:
        print(
            f"\nValidation metric {final_qwk} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
