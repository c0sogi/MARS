import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, compute_metrics
from library.dataset import PearsonDataset, DataCollator, prepare_data
from library.model import CustomModel
from library.awp import AWP
from library.engine import get_optimizer_params, train_fn, valid_fn


def run():
    # 1. Setup and Configuration
    # Adjust Config for fast baseline execution on A100
    Config.epochs = 3
    Config.train_batch_size = 16  # Increase batch size for A100
    Config.valid_batch_size = 32
    Config.num_workers = 4

    # Ensure reproducibility
    seed_everything(Config.seed)

    # Logger
    logger = get_logger(log_filename="run.log")
    logger.info("Starting runfile.py execution...")
    logger.info(f"Device: {Config.device}")

    # 2. Data Loading
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    logger.info("Preparing data...")
    df_train, df_val, df_test = prepare_data(load_cached_data=True)

    # Create Datasets
    train_dataset = PearsonDataset(df_train, tokenizer, is_test=False)
    val_dataset = PearsonDataset(df_val, tokenizer, is_test=False)

    # Create Collator
    collator = DataCollator(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    logger.info("Initializing model...")
    model = CustomModel()
    model.to(Config.device)

    # Optimizer with LLRD
    optimizer_params = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate * 5,  # Higher LR for heads
        weight_decay=Config.weight_decay,
    )
    optimizer = AdamW(optimizer_params, lr=Config.learning_rate, eps=Config.eps)

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        num_cycles=Config.num_cycles,
    )

    # AWP
    awp = (
        AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)
        if Config.use_awp
        else None
    )

    # 4. Training Loop
    best_score = -1.0
    model_save_path = os.path.join(Config.output_dir, "best_model.pth")

    logger.info(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, optimizer, epoch, scheduler, Config.device, awp
        )

        # Validate
        val_loss, val_score = valid_fn(valid_loader, model, Config.device)

        logger.info(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pearson: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            logger.info(f"New best score! Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

    # 5. Final Validation & Failure Analysis
    logger.info("Loading best model for analysis...")
    model.load_state_dict(torch.load(model_save_path, map_location=Config.device))
    model.eval()

    # Generate predictions on validation set for analysis
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            targets = batch["score"].to(Config.device)

            outputs = model(input_ids, attention_mask)
            scores = outputs["score"].view(-1)

            val_preds.extend(scores.cpu().numpy())
            val_labels.extend(targets.cpu().numpy())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Calculate Final Metric
    final_metrics = compute_metrics(val_preds, val_labels)
    final_pearson = final_metrics["pearson"]

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_pearson}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    df_val["prediction"] = val_preds
    df_val["abs_error"] = (df_val["score"] - df_val["prediction"]).abs()

    # Feature Engineering for Analysis
    df_val["anchor_len"] = df_val["anchor"].astype(str).apply(len)
    df_val["target_len"] = df_val["target"].astype(str).apply(len)

    def get_jaccard(row):
        set_a = set(str(row["anchor"]).lower().split())
        set_b = set(str(row["target"]).lower().split())
        union = len(set_a.union(set_b))
        return len(set_a.intersection(set_b)) / union if union > 0 else 0.0

    df_val["jaccard"] = df_val.apply(get_jaccard, axis=1)

    # Compute Correlations
    analysis_features = ["anchor_len", "target_len", "jaccard"]
    print("\nCorrelation between Absolute Error and Input Features:")
    for feat in analysis_features:
        if df_val[feat].std() > 0:
            corr, _ = scipy.stats.pearsonr(df_val["abs_error"], df_val[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: NaN (Constant feature)")

    # 6. Submission Generation
    THRESHOLD = 0.8513218760490417

    if final_pearson > THRESHOLD:
        logger.info(
            f"Validation score {final_pearson} > {THRESHOLD}. Generating submission..."
        )

        test_dataset = PearsonDataset(df_test, tokenizer, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)

                outputs = model(input_ids, attention_mask)
                scores = outputs["score"].view(-1)
                test_preds.extend(scores.cpu().numpy())

        # Post-processing: Clip to [0, 1]
        test_preds = np.clip(test_preds, 0, 1)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": df_test["id"], "score": test_preds})

        # Ensure output directory exists
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.warning(
            f"Validation score {final_pearson} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
