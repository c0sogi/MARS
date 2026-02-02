import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_logger, compute_metrics
from library.dataset import load_dataset, ChatbotArenaDataset
from library.model import SiameseDebertaModel
from library.engine import train_one_epoch, validate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("main")
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Limit training data for fast baseline requirement
    # 10,000 samples is enough to learn a decent signal > random guess (1.09)
    TRAIN_LIMIT = 10000

    logger.info(f"Loading Training Data (Limit={TRAIN_LIMIT})...")
    train_dataset = load_dataset(
        split="train", tokenizer=tokenizer, load_cached_data=True, limit=TRAIN_LIMIT
    )

    logger.info("Loading Validation Data (Full)...")
    val_dataset = load_dataset(
        split="val", tokenizer=tokenizer, load_cached_data=True, limit=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = SiameseDebertaModel()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=Config.USE_FP16)

    # 4. Training Loop
    best_val_loss = float("inf")

    logger.info("Starting Training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(
                f"Validation Loss Improved to {best_val_loss:.4f}. Saving model..."
            )
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            logger.info(f"Validation Loss {val_loss:.4f} did not improve.")

    # 5. Final Evaluation & Failure Analysis
    logger.info("Loading best model for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Re-run validation to get predictions for analysis
    logger.info("Running final validation inference...")
    # We need raw predictions and targets
    all_preds = []
    all_targets = []

    # Simple inference loop for analysis data collection
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            response_mask = batch["response_mask"].to(device)
            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_FP16):
                logits = model(input_ids, attention_mask, response_mask, scalars)

            probs = torch.softmax(logits.float(), dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metrics = compute_metrics(all_targets, all_preds)
    final_log_loss = final_metrics["log_loss"]

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_log_loss}")

    # --- Failure Analysis ---
    logger.info("Performing Failure Analysis...")

    # Calculate Cross Entropy Loss per sample
    # Loss = - sum(y_true * log(y_pred))
    # Clip preds to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)
    sample_losses = -np.sum(all_targets * np.log(preds_clipped), axis=1)

    # Get Scalar Features from dataset (df is accessible via dataset.df)
    # The dataset stores scalars as a list of lists in the 'scalars' column of the dataframe
    # Format: [log(prompt), log(resp_a), log(resp_b)]
    val_df = val_dataset.df
    scalars_list = val_df["scalars"].tolist()
    scalars_arr = np.array(scalars_list)

    prompt_lens = scalars_arr[:, 0]
    resp_a_lens = scalars_arr[:, 1]
    resp_b_lens = scalars_arr[:, 2]

    # Calculate correlations
    corr_prompt = np.corrcoef(sample_losses, prompt_lens)[0, 1]
    corr_resp_a = np.corrcoef(sample_losses, resp_a_lens)[0, 1]
    corr_resp_b = np.corrcoef(sample_losses, resp_b_lens)[0, 1]

    print("Correlation between Error (Log Loss) and Input Features:")
    print(f"Log(Prompt Length): {corr_prompt:.4f}")
    print(f"Log(Response A Length): {corr_resp_a:.4f}")
    print(f"Log(Response B Length): {corr_resp_b:.4f}")

    # 6. Submission
    TARGET_METRIC = 1.010114098334317

    if final_log_loss < TARGET_METRIC:
        logger.info(
            f"Metric {final_log_loss} < {TARGET_METRIC}. Generating submission..."
        )

        # Load Test Data
        test_dataset = load_dataset(
            split="test", tokenizer=tokenizer, load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Predict
        test_probs = predict(model, test_loader, device)

        # Create Submission DataFrame
        # Test dataset df has 'id'
        sub_df = pd.DataFrame(
            {
                "id": test_dataset.df["id"].values,
                "winner_model_a": test_probs[:, 0],
                "winner_model_b": test_probs[:, 1],
                "winner_tie": test_probs[:, 2],
            }
        )

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Metric {final_log_loss} >= {TARGET_METRIC}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
