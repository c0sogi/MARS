import os
import gc
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    get_one_cycle_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)

from library.config import Config
from library.utils import seed_everything, get_score
from library.data import (
    load_dataset_from_metadata,
    get_tokenizer,
    ToxicDataset,
    MLMDataset,
    prepare_test_loader,
)
from library.model import CustomDeberta
from library.engine import train_mlm, train_fn, valid_fn, inference_fn
from library.awp import AWP

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# We modify the Config to ensure the code completes within the 2-hour limit.
Config.epochs = 2
Config.mlm_epochs = 1
Config.n_folds = 1  # Single split training for baseline speed
Config.debug = False  # We handle subsampling manually
MAX_TRAIN_SAMPLES = 20000
MAX_MLM_SAMPLES = 40000


def run():
    # 1. Setup
    print("Initializing...")
    seed_everything(Config.seed)
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    tokenizer = get_tokenizer()

    # 2. Data Loading
    print("Loading data...")
    # Load Train and Val separately to strictly follow "hold-out validation set" requirement
    train_df = load_dataset_from_metadata(
        Config.train_meta_path,
        Config.train_raw_path,
        load_cached_data=True,
        cache_name="train_split_only",
    )
    val_df = load_dataset_from_metadata(
        Config.val_meta_path,
        Config.train_raw_path,
        load_cached_data=True,
        cache_name="val_split_only",
    )

    # Load Test Data (for MLM and Submission)
    test_df = load_dataset_from_metadata(
        Config.test_meta_path,
        Config.test_raw_path,
        load_cached_data=True,
        cache_name="test_full",
    )

    # Subsample Training Data for Speed
    print(f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples...")
    train_df = train_df.sample(
        n=min(len(train_df), MAX_TRAIN_SAMPLES), random_state=Config.seed
    ).reset_index(drop=True)

    # 3. Domain-Adaptive Pre-training (MLM)
    print("Preparing MLM data...")
    # Combine texts and subsample
    train_texts = train_df["comment_text"].values
    test_texts = test_df["comment_text"].values
    all_texts = np.concatenate([train_texts, test_texts])

    if len(all_texts) > MAX_MLM_SAMPLES:
        np.random.seed(Config.seed)
        all_texts = np.random.choice(all_texts, MAX_MLM_SAMPLES, replace=False)

    mlm_dataset = MLMDataset(all_texts, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
    )
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=Config.mlm_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # Run MLM
    train_mlm(mlm_loader, Config.device)

    # Cleanup MLM
    del mlm_loader, mlm_dataset, all_texts
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Supervised Training
    print("Starting Supervised Training...")

    # Prepare DataLoaders
    train_dataset = ToxicDataset(train_df, tokenizer)
    val_dataset = ToxicDataset(val_df, tokenizer)  # Full validation set

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize Model with MLM weights
    model = CustomDeberta(pretrained_path=Config.mlm_model_dir)
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_one_cycle_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.pct_start * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )

    best_score = -1.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    for epoch in range(Config.epochs):
        train_loss = train_fn(
            train_loader, model, None, optimizer, scheduler, Config.device, epoch, awp
        )
        val_loss, val_score = valid_fn(val_loader, model, Config.device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 5. Validation Analysis
    print("Performing Validation Analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.to(Config.device)
    model.eval()

    # Get predictions on full validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()

            val_preds.append(probs)
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Calculate Final Metric
    final_metric = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error and length
    errors = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Calculate word counts for validation set
    # Note: val_df order matches val_loader order (shuffle=False)
    val_word_counts = val_df["comment_text"].apply(lambda x: len(str(x).split())).values

    correlation = np.corrcoef(errors, val_word_counts)[0, 1]
    print(f"Correlation between error magnitude and word count: {correlation:.4f}")

    # 6. Submission
    THRESHOLD = 0.9920879090652149

    if final_metric > THRESHOLD:
        print("Metric threshold met. Generating submission...")

        test_loader = prepare_test_loader(test_df, tokenizer)
        test_preds = inference_fn(test_loader, model, Config.device)

        submission = pd.DataFrame(test_preds, columns=Config.target_cols)
        submission["id"] = test_df["id"]

        # Reorder columns
        cols = ["id"] + Config.target_cols
        submission = submission[cols]

        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Metric {final_metric:.6f} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
