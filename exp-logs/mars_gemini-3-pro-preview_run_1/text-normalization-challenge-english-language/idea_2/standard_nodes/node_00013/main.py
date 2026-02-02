import sys
import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_dataloaders
from library.engine import train_model
from library.inference import Normalizer, generate_submission
from library.model import LSTMTagger


def main():
    # 1. Setup & Configuration
    set_seed(Config.SEED)
    device = get_device()

    # Configure for fast baseline execution while maintaining high performance
    # Using full dataset ensures Knowledge Base quality and Model generalization.
    Config.NUM_EPOCHS = 5
    Config.DEBUG = False

    # 2. Data Loading
    # Load cached data to save time
    (
        train_loader,
        val_loader,
        test_loader,
        vocab_tokens,
        vocab_classes,
        knowledge_base,
    ) = get_dataloaders(load_cached_data=True)

    # 3. Training
    # train_model handles the training loop and saves the best model to Config.MODEL_SAVE_PATH
    _ = train_model(train_loader, val_loader, vocab_tokens, vocab_classes)

    # 4. Validation
    # We need to compute the exact string match accuracy for the task metric.
    # Load the best model saved during training.
    model = LSTMTagger(
        vocab_size=len(vocab_tokens),
        num_classes=len(vocab_classes),
        pad_token_id=vocab_tokens.stoi.get(Config.PAD_TOKEN, 0),
    )

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model file not found.")
        return

    model.to(device)
    model.eval()

    normalizer = Normalizer(knowledge_base)

    total_tokens = 0
    correct_tokens = 0

    # Lists for failure analysis
    errors = []
    token_lengths = []

    # Access the underlying dataframe for ground truth
    val_df = val_loader.dataset.df
    current_df_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)

            # Forward pass
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            # Batch processing
            batch_size = input_ids.size(0)

            # Transpose batch lists to iterate sample-wise
            raw_tokens_T = batch["raw_tokens"]
            batch_raw_tokens = list(zip(*raw_tokens_T))

            # Get ground truth rows
            end_idx = current_df_idx + batch_size
            df_rows = val_df.iloc[current_df_idx:end_idx]
            current_df_idx = end_idx

            # Iterate through each sentence in the batch
            for i, row in enumerate(df_rows.itertuples(index=False)):
                gt_after_seq = row.after

                # Get sequences for this sample
                pred_seq_indices = preds[i]
                raw_token_seq = batch_raw_tokens[i]

                # Determine evaluation length (handle truncation)
                seq_len = len(gt_after_seq)
                eval_len = min(seq_len, Config.MAX_LEN)

                for t_idx in range(eval_len):
                    token = raw_token_seq[t_idx]
                    class_idx = pred_seq_indices[t_idx]
                    pred_class = vocab_classes.lookup_token(class_idx)

                    # Normalize
                    pred_text = normalizer.normalize(token, pred_class)
                    gt_text = gt_after_seq[t_idx]

                    # Check correctness
                    is_correct = pred_text == gt_text

                    total_tokens += 1
                    if is_correct:
                        correct_tokens += 1

                    # Log for failure analysis
                    errors.append(0 if is_correct else 1)
                    token_lengths.append(len(token))

    final_metric = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 5. Failure Analysis
    if len(errors) > 0:
        # Calculate correlation between Error (0/1) and Token Length
        # Using numpy for efficiency
        corr = np.corrcoef(errors, token_lengths)[0, 1]
        print(f"Correlation between Error and Token Length: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 0.9861543320467205
    if final_metric > THRESHOLD:
        generate_submission(debug=False, load_cached_data=True)
    else:
        print(
            f"Validation metric {final_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
