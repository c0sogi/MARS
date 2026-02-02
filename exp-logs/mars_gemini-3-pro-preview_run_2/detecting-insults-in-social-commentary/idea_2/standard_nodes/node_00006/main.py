import os
import sys
import copy
import torch
import pandas as pd
import numpy as np
import random
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

# Import from library
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import TransformerClassifier
from library.train_utils import train_one_epoch, evaluate, predict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    transformers.logging.set_verbosity_error()

    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Using load_cached_data=True as requested
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = TransformerClassifier(model_name=Config.MODEL_NAME)
    model.to(Config.DEVICE)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    loss_fn = BCEWithLogitsLoss()

    # 4. Training Loop
    print("Starting training...")
    best_auc = 0.0
    best_model_state = None

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, loss_fn
        )
        val_loss, val_auc = evaluate(model, val_loader, Config.DEVICE, loss_fn)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            # Deep copy the state dict to ensure we save the exact weights
            best_model_state = copy.deepcopy(model.state_dict())

    # Load best model for final evaluation and inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save the best model to disk as per config
        torch.save(best_model_state, Config.MODEL_SAVE_PATH)
        print(f"Best model saved with AUC: {best_auc:.4f}")

    # 5. Final Validation and Failure Analysis
    print("\nRunning Failure Analysis...")

    # Re-evaluate on validation set to get predictions
    # We need raw probabilities for analysis
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            val_preds.extend(probs.cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Calculate Final Metric
    try:
        from sklearn.metrics import roc_auc_score

        final_auc = roc_auc_score(val_labels, val_preds)
    except ValueError:
        final_auc = 0.0

    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Input Length
    # Load validation metadata to get text features
    df_val = pd.read_csv(Config.VAL_FILE)

    # Calculate error magnitude
    errors = np.abs(val_labels - val_preds)

    # Extract features
    # Note: We need to ensure the order matches. DataLoader is sequential for validation (shuffle=False),
    # so the order of df_val should match val_preds.

    # Helper to clean text for length calculation (same logic as data_utils roughly)
    def get_text_len(text):
        return len(str(text))

    def get_word_count(text):
        return len(str(text).split())

    df_val["char_len"] = df_val["Comment"].apply(get_text_len)
    df_val["word_len"] = df_val["Comment"].apply(get_word_count)

    # Calculate correlations
    corr_char = np.corrcoef(errors, df_val["char_len"])[0, 1]
    corr_word = np.corrcoef(errors, df_val["word_len"])[0, 1]

    print("-" * 30)
    print("Failure Analysis (Correlation with Error Magnitude):")
    print(f"Character Length Correlation: {corr_char:.4f}")
    print(f"Word Count Correlation:       {corr_word:.4f}")
    print("-" * 30)

    # 6. Submission
    THRESHOLD = 0.942
    if final_auc > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")

        # Generate predictions for test set
        test_probs = predict(model, test_loader, Config.DEVICE)

        # Load test metadata to construct submission
        df_test = pd.read_csv(Config.TEST_FILE)

        # Ensure lengths match
        if len(df_test) != len(test_probs):
            print(
                f"Warning: Mismatch in test set length. Metadata: {len(df_test)}, Preds: {len(test_probs)}"
            )
            # Truncate or pad is risky, but usually they match if data loading is correct.
            # We assume they match based on the provided library code.

        # Update Insult column
        df_test["Insult"] = test_probs

        # Save submission
        df_test.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"Validation metric {final_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
