import os
import sys
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer
from torch.optim import AdamW

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_test_loader
from library.model import TweetModel
from library.engine import train_fn, eval_fn


def run_demonstration():
    # 1. Setup and Configuration
    print("\n--- 1. Setup and Configuration ---")
    # Override Config for speed in this demonstration
    # We use a smaller model (DistilRoBERTa) instead of DeBERTa-Large to ensure
    # the demo runs quickly within the environment's constraints.
    demo_model = "distilroberta-base"

    # Update Config class attributes directly before setup
    Config.MODEL_PATH = demo_model
    Config.TOKENIZER_PATH = demo_model

    # Initialize Config with debug mode enabled (uses small data subset)
    Config.setup(debug=True, epochs=1, batch_size=8)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Suppress verbose transformers logging
    transformers.logging.set_verbosity_error()

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("\n--- 2. Data Preparation ---")
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load DataLoaders (Debug mode loads a small subset of train/val)
    # We set load_cached_data=False to demonstrate processing logic
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        batch_size=Config.TRAIN_BATCH_SIZE,
        load_cached_data=False,
        debug=Config.DEBUG,
    )

    # Verify Data Integrity
    print("Verifying DataLoader batch structure...")
    sample_batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_labels",
        "end_labels",
        "content_masks",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Missing key {key} in batch"
        assert torch.is_tensor(sample_batch[key]), f"{key} is not a tensor"

    # Check shapes (Batch Size, Seq Len)
    b_size, seq_len = sample_batch["input_ids"].shape
    assert b_size <= Config.TRAIN_BATCH_SIZE
    assert seq_len == Config.MAX_LEN
    print("Data validation passed.")

    # 3. Model Initialization
    print("\n--- 3. Model Initialization ---")
    model = TweetModel(model_path=Config.MODEL_PATH)
    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    print("\n--- 4. Training ---")
    # Run for 1 epoch as configured
    avg_train_loss = train_fn(train_loader, model, optimizer, device)
    print(f"Epoch 1/{Config.EPOCHS} - Training Loss: {avg_train_loss:.4f}")

    # 5. Evaluation
    print("\n--- 5. Evaluation ---")
    val_loss, val_jaccard = eval_fn(val_loader, model, device, tokenizer)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Jaccard Score: {val_jaccard:.4f}")

    # 6. Inference on Test Set
    print("\n--- 6. Inference on Test Set ---")
    test_loader, test_df = get_test_loader(
        tokenizer, batch_size=Config.VALID_BATCH_SIZE, load_cached_data=False
    )

    model.eval()
    predictions = []

    # Inference Loop (Simplified version of eval_fn logic adapted for test without labels)
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            start_logits, end_logits, content_logits = model(input_ids, attention_mask)

            # Decode probabilities
            p_start = torch.softmax(start_logits, dim=1).cpu().numpy()
            p_end = torch.softmax(end_logits, dim=1).cpu().numpy()
            p_content = torch.sigmoid(content_logits).cpu().numpy()

            ids = input_ids.cpu().numpy()

            # Hybrid Decoding Strategy
            for i in range(len(ids)):
                ps = p_start[i]
                pe = p_end[i]
                pc = p_content[i]

                best_score = -float("inf")
                best_start = 0
                best_end = 0
                seq_len_curr = len(ps)

                # Span search
                for s_idx in range(seq_len_curr):
                    if ps[s_idx] < 0.001:
                        continue
                    for e_idx in range(s_idx, seq_len_curr):
                        if pe[e_idx] < 0.001:
                            continue

                        content_score = np.mean(pc[s_idx : e_idx + 1])
                        score = ps[s_idx] + pe[e_idx] + content_score

                        if score > best_score:
                            best_score = score
                            best_start = s_idx
                            best_end = e_idx

                # Decode token IDs to string
                pred_ids = ids[i][best_start : best_end + 1]
                pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)
                predictions.append(pred_str)

    # 7. Submission Generation
    print("\n--- 7. Submission Generation ---")
    # Ensure predictions match test set length
    assert len(predictions) == len(
        test_df
    ), f"Prediction count {len(predictions)} != Test set size {len(test_df)}"

    submission = pd.DataFrame(
        {
            "textID": test_df["textID"],
            "selected_text": predictions,
            "text": test_df["text"],
        }
    )

    # Post-processing: If prediction is empty (rare), fallback to full text
    # Also handle the specific requirement: "selected text needs to be quoted"
    # The submission format example shows: 2,"very good"
    # Pandas to_csv handles quoting automatically if we configure it,
    # but the requirement says 'selected_text' column content should be the string.
    # The example `2,"very good"` implies standard CSV quoting for strings containing spaces.

    # Fallback for empty predictions
    submission["selected_text"] = submission.apply(
        lambda row: (
            row["selected_text"]
            if len(str(row["selected_text"]).strip()) > 0
            else row["text"]
        ),
        axis=1,
    )

    # Drop the temporary 'text' column to match submission format
    submission = submission.drop(columns=["text"])

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    # Verify Submission File
    print("Verifying submission file...")
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Columns: {list(df_sub.columns)}")

    assert "textID" in df_sub.columns
    assert "selected_text" in df_sub.columns
    assert len(df_sub) == len(test_df)

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
