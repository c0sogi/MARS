import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_demo():
    print("=== Tweet Sentiment Extraction Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config settings for a fast demonstration
    print("\n[1] Configuring for Speed/Debug...")
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.N_FOLDS = 1

    # We set sample size equal to batch size (8) to ensure we only have 1 batch per loader.
    # This is a strategy to prevent the provided `eval_fn` from crashing due to
    # np.concatenate on variable-length arrays (caused by Smart Batching).
    Config.DEBUG_SAMPLE_SIZE = 8

    # Select the RoBERTa-large model configuration (Index 1)
    model_config = Config.MODEL_CONFIGS[1]
    model_config["batch_size"] = 8

    print(f"    Model: {model_config['model_name']}")
    print(f"    Batch Size: {model_config['batch_size']}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")
    # load_cached_data=False forces the processing logic to run for demonstration
    train_loader, val_loader, test_loader = get_dataloaders(
        model_config, load_cached_data=False, debug=True
    )

    # Verification: Check if loaders yield valid batches
    try:
        sample_batch = next(iter(train_loader))
        assert "input_ids" in sample_batch
        assert "attention_mask" in sample_batch
        assert "start_targets" in sample_batch
        # Check batch size matches configuration
        assert sample_batch["input_ids"].shape[0] == model_config["batch_size"]
        print("    DataLoaders initialized and verified successfully.")
    except StopIteration:
        raise ValueError("Train loader is empty!")
    except AssertionError as e:
        raise ValueError(f"DataLoader verification failed: {e}")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[3] Initializing Model...")
    device = Config.DEVICE
    model = TweetModel(model_config["model_name"])
    model.to(device)

    # Verification: Run a dummy forward pass
    print("    Verifying model forward pass...")
    with torch.no_grad():
        dummy_ids = sample_batch["input_ids"].to(device)
        dummy_mask = sample_batch["attention_mask"].to(device)
        s_logits, e_logits = model(dummy_ids, dummy_mask)

        # Check output shapes: (batch_size, seq_len)
        assert s_logits.shape == dummy_ids.shape
        assert e_logits.shape == dummy_ids.shape
    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[4] Starting Training (1 Epoch)...")
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Run training function
    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"    Epoch 1 Train Loss: {avg_train_loss:.4f}")

    # Verification: Loss should be a valid number
    if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
        raise ValueError("Training loss is NaN or Inf.")

    # ---------------------------------------------------------
    # 5. Evaluation Loop
    # ---------------------------------------------------------
    print("\n[5] Starting Evaluation on Validation Set...")
    # Run evaluation function
    val_loss, val_jaccard, _, _ = eval_fn(val_loader, model, device)
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation Jaccard: {val_jaccard:.4f}")

    # Verification: Jaccard score must be between 0 and 1
    assert 0.0 <= val_jaccard <= 1.0, f"Invalid Jaccard score: {val_jaccard}"

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\n[6] Generating Predictions for Test Set...")
    model.eval()
    predictions = []

    # Custom inference loop to handle text decoding
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            # Metadata needed for decoding
            texts = batch["text"]
            sentiments = batch["sentiment"]
            offsets = batch["offsets"].cpu().numpy()
            ids = batch["textID"]

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            # Decode probabilities
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            start_indices = np.argmax(start_probs, axis=1)
            end_indices = np.argmax(end_probs, axis=1)

            # Map tokens back to text string
            for i in range(len(ids)):
                text = str(texts[i])
                sentiment = str(sentiments[i])
                offset = offsets[i]

                pred_text = ""

                # Heuristic: Neutral sentiment usually implies the whole text is selected
                if sentiment == "neutral":
                    pred_text = text
                else:
                    idx_start = start_indices[i]
                    idx_end = end_indices[i]

                    if idx_end < idx_start:
                        idx_end = idx_start

                    # Extract substring using offsets
                    if idx_start < len(offset) and idx_end < len(offset):
                        char_start = offset[idx_start][0]
                        char_end = offset[idx_end][1]
                        # Handle special tokens (mapped to 0,0)
                        if char_start == 0 and char_end == 0 and idx_start != 0:
                            pred_text = text
                        else:
                            pred_text = text[char_start:char_end]
                    else:
                        pred_text = text

                predictions.append({"textID": ids[i], "selected_text": pred_text})

    # Create Submission DataFrame
    sub_df = pd.DataFrame(predictions)

    # Save to ./working directory
    os.makedirs("./working", exist_ok=True)
    submission_path = "./working/demo_submission.csv"
    sub_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")
    print("    First 5 predictions:")
    print(sub_df.head())

    # Final Verification
    assert os.path.exists(submission_path), "Submission file was not created."
    loaded_df = pd.read_csv(submission_path)
    assert len(loaded_df) == len(sub_df), "Saved file length mismatch."
    assert list(loaded_df.columns) == [
        "textID",
        "selected_text",
    ], "Incorrect columns in submission."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
