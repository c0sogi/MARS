import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer

# Append current directory to system path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data import get_dataloaders
from library.model import DistilRoBERTaDualEncoder
from library.engine import get_optimizer, get_scheduler, train_fn, eval_fn, inference_fn
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing Tokenizer and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.BACKBONE)

    # Load data using cached files if available
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer, load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DistilRoBERTaDualEncoder()
    model.to(device)

    # 4. Optimizer and Scheduler
    # Calculate total training steps
    num_train_steps = len(train_loader) * Config.EPOCHS
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_train_steps)

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = -1.0

    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.EPOCHS} ---")

        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_score = eval_fn(val_loader, model, device)
        # Note: eval_fn prints the validation metrics

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with score: {val_score:.4f}")

    # 6. Final Evaluation & Metric Printing
    print(f"Final Validation Metric: {best_score}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # Generate predictions on validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for data in val_loader:
            q_input_ids = data["q_input_ids"].to(device)
            q_attention_mask = data["q_attention_mask"].to(device)
            a_input_ids = data["a_input_ids"].to(device)
            a_attention_mask = data["a_attention_mask"].to(device)
            labels = data["labels"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            batch_preds = torch.sigmoid(logits).cpu().numpy()

            val_preds.append(batch_preds)
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Mean Absolute Error (MAE) per sample
    # Shape: (N_samples,)
    mae_per_sample = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Load validation metadata to get features
    val_df = pd.read_csv(Config.VAL_PATH)

    # Feature 1: Question Length (Title + Body)
    val_df["q_len"] = (
        val_df["question_title"].fillna("") + " " + val_df["question_body"].fillna("")
    ).str.len()
    # Feature 2: Answer Length
    val_df["a_len"] = val_df["answer"].fillna("").str.len()

    # Calculate correlations
    corr_q = np.corrcoef(mae_per_sample, val_df["q_len"])[0, 1]
    corr_a = np.corrcoef(mae_per_sample, val_df["a_len"])[0, 1]

    print(f"Correlation between Error and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length: {corr_a:.4f}")

    # 8. Submission Generation
    THRESHOLD = 0.40802662717842303

    if best_score > THRESHOLD:
        print(
            f"\nValidation score ({best_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test Set
        test_preds = inference_fn(test_loader, model, device)

        # Prepare Submission DataFrame
        test_df = pd.read_csv(Config.TEST_PATH)
        submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)

        # Insert qa_id at the beginning
        submission.insert(0, "qa_id", test_df["qa_id"])

        # Ensure output directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Save
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation score ({best_score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
