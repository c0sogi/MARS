import os
import numpy as np
import pandas as pd
import torch
from transformers import get_cosine_schedule_with_warmup

# Import from library files
from library.config import Config
from library.utils import seed_everything, AWP, jaccard
from library.data import get_dataloaders
from library.model import TweetModel
from library.engine import (
    train_fn,
    eval_fn,
    predict_fn,
    get_optimizer_params,
    decode_prediction,
)


def run_pipeline():
    # 1. Configuration
    config = Config()

    # Set seed for reproducibility
    seed_everything(config.SEED)

    print("--- Starting Pipeline ---")
    print(f"Device: {config.DEVICE}")

    # 2. Data Loading
    print("Loading Data...")
    # Using cached data to speed up loading
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Setup
    print("Initializing Model...")
    model = TweetModel(config)
    model.to(config.DEVICE)

    # 4. Optimizer & Scheduler
    # Apply Layer-wise Learning Rate Decay (LLRD)
    optimizer_params = get_optimizer_params(model, config)
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=config.LEARNING_RATE, eps=config.eps, betas=config.betas
    )

    num_train_steps = int(len(train_loader) * config.EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        num_training_steps=num_train_steps,
    )

    # 5. Adversarial Weight Perturbation (AWP)
    awp = AWP(model, optimizer, adv_lr=config.AWP_LR, adv_eps=config.AWP_EPS)

    # 6. Training Loop
    best_jaccard = 0.0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.bin")

    # Remove existing model file to ensure we save a new one
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    print("Starting Training...")
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        # Train one epoch
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, awp, epoch, config, config.DEVICE
        )

        # Evaluate
        val_jaccard = eval_fn(val_loader, model, config, config.DEVICE)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
        )

        # Save best model
        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training Finished.")

    # 7. Validation Assessment & Failure Analysis
    print("Performing Validation Assessment & Failure Analysis...")

    # Load best model for analysis
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    else:
        print("Warning: No best model saved. Using current model state.")

    model.eval()

    val_scores = []
    val_lengths = []

    # Manual evaluation loop to collect per-sample metrics
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            offsets = batch["offsets"].numpy()
            raw_texts = batch["raw_text"]
            sentiments = batch["sentiment"]
            selected_texts = batch["selected_text"]

            start_logits, end_logits, _ = model(input_ids, attention_mask)

            for i in range(len(input_ids)):
                pred_text = decode_prediction(
                    start_logits[i],
                    end_logits[i],
                    raw_texts[i],
                    offsets[i],
                    sentiments[i],
                )

                score = jaccard(selected_texts[i], pred_text)
                val_scores.append(score)
                val_lengths.append(len(raw_texts[i]))

    final_metric = np.mean(val_scores)
    # Print required metric with full precision
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between error and text length
    # Error is defined as (1 - Jaccard Score)
    errors = 1.0 - np.array(val_scores)
    lengths = np.array(val_lengths)

    df_analysis = pd.DataFrame({"error": errors, "length": lengths})

    if len(df_analysis) > 1:
        corr_len = df_analysis["error"].corr(df_analysis["length"])
        print(f"Correlation (Error vs Text Length): {corr_len:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 8. Submission Logic
    THRESHOLD = 0.7043342108129372

    if final_metric > THRESHOLD:
        print(
            f"Validation metric {final_metric} > {THRESHOLD}. Generating submission..."
        )

        # Generate predictions on test set
        test_preds = predict_fn(test_loader, model, config, config.DEVICE)

        # Load test metadata to get IDs (order is preserved in DataLoader)
        test_df = pd.read_csv(config.TEST_META_PATH)
        if config.DEBUG_SAMPLE_SIZE:
            test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

        submission = pd.DataFrame(
            {"textID": test_df["textID"], "selected_text": test_preds}
        )

        submission.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE}")
    else:
        print(f"Validation metric {final_metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run_pipeline()
