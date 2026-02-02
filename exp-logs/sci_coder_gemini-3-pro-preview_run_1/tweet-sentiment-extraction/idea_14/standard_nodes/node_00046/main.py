import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from library.config import Config
from library.data import get_loaders, get_test_loader
from library.model import TweetModel
from library.loss import SoftTargetKLLoss, RDropLoss
from library.engine import train_fn, eval_fn
from library.utils import seed_everything, get_optimizer_params, jaccard, normalize_text


def run():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Tokenizer and Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Loaders return data with neutrals filtered out for training/val
    train_loader, val_loader = get_loaders(tokenizer, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = TweetModel(Config)
    model.to(device)

    # 4. Optimization
    # Use LLRD (Layer-wise Learning Rate Decay)
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.lr,
        decoder_lr=Config.lr * 5,  # Higher LR for head
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )
    optimizer = AdamW(optimizer_parameters, lr=Config.lr, eps=Config.eps)

    # Scheduler
    # Limit epochs for fast baseline execution
    num_epochs = 2
    num_train_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Losses
    criterion_task = SoftTargetKLLoss()
    criterion_rdrop = RDropLoss()

    # 5. Training Loop
    print(f"Starting Training for {num_epochs} epochs...")
    best_jaccard = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.bin")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            device,
            scheduler,
            criterion_task,
            criterion_rdrop,
            Config,
        )

        # Validate (on non-neutral subset)
        val_loss, val_jaccard = eval_fn(
            val_loader, model, device, criterion_task, Config
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Jaccard (Non-Neutral): {val_jaccard:.4f}"
        )

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Jaccard: {best_jaccard:.4f}")

    # 6. Final Validation Metric Calculation
    print("\nComputing Final Validation Metric...")
    # Load full validation metadata to include neutrals
    val_meta_df = pd.read_csv(Config.VAL_META_PATH)

    # Split into neutral and non-neutral
    neutral_df = val_meta_df[val_meta_df["sentiment"] == "neutral"]
    non_neutral_df = val_meta_df[val_meta_df["sentiment"] != "neutral"]

    # Calculate Neutral Jaccard (Identity Mapping)
    # Neutrals: selected_text is usually the text itself.
    # We calculate it explicitly to be precise.
    neutral_jaccards = []
    for _, row in neutral_df.iterrows():
        # Handle potential NaNs by converting to string
        t = str(row["text"])
        st = str(row["selected_text"])
        neutral_jaccards.append(jaccard(t, st))

    avg_neutral_jaccard = np.mean(neutral_jaccards) if neutral_jaccards else 0.0

    # Get best non-neutral jaccard (from training loop)
    # Note: Ideally we would re-run inference on the best model to be exact,
    # but using the tracked best_jaccard is sufficient for this baseline logic
    # and saves time.
    avg_non_neutral_jaccard = best_jaccard

    # Weighted Average
    n_neutral = len(neutral_df)
    n_non_neutral = len(non_neutral_df)
    total_samples = n_neutral + n_non_neutral

    final_metric = (
        avg_neutral_jaccard * n_neutral + avg_non_neutral_jaccard * n_non_neutral
    ) / total_samples

    print(f"Neutral Jaccard: {avg_neutral_jaccard:.4f} (Count: {n_neutral})")
    print(
        f"Non-Neutral Jaccard: {avg_non_neutral_jaccard:.4f} (Count: {n_non_neutral})"
    )
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # We need per-sample errors on the non-neutral set.
    # We will reload the best model and run inference on val_loader one last time.
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_errors = []
    val_text_lens = []

    # We need to align predictions with metadata.
    # eval_fn computes average, we need individual scores.
    # We'll implement a quick inference loop here similar to eval_fn but storing data.

    # Re-read non-neutral validation data to ensure alignment with loader
    # The loader iterates sequentially.
    val_meta_non_neutral = pd.read_csv(Config.VAL_META_PATH)
    val_meta_non_neutral = val_meta_non_neutral[
        val_meta_non_neutral["sentiment"] != "neutral"
    ].reset_index(drop=True)

    current_idx = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offsets = batch["offsets"].numpy()
            texts = batch["text"]

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            batch_size = input_ids.size(0)

            for i in range(batch_size):
                if current_idx >= len(val_meta_non_neutral):
                    break

                target_text = str(
                    val_meta_non_neutral.iloc[current_idx]["selected_text"]
                )
                original_text = texts[i]
                offset = offsets[i]

                # Decoding
                start_p = start_probs[i]
                end_p = end_probs[i]
                score_mat = np.outer(start_p, end_p)
                score_mat = np.triu(score_mat)
                best_idx = np.argmax(score_mat)
                best_start_idx, best_end_idx = np.unravel_index(
                    best_idx, score_mat.shape
                )

                if best_start_idx >= len(offset) or best_end_idx >= len(offset):
                    pred_text = original_text
                else:
                    char_start = offset[best_start_idx][0]
                    char_end = offset[best_end_idx][1]
                    if char_start == 0 and char_end == 0:
                        pred_text = original_text
                    else:
                        pred_text = original_text[char_start:char_end]

                score = jaccard(target_text, pred_text)
                error = 1.0 - score

                val_errors.append(error)
                val_text_lens.append(len(str(original_text)))

                current_idx += 1

    # Compute correlation
    if len(val_errors) > 0:
        correlation = np.corrcoef(val_errors, val_text_lens)[0, 1]
        print(
            f"Correlation between Error (1-Jaccard) and Text Length: {correlation:.4f}"
        )
    else:
        print("Not enough data for failure analysis.")

    # 8. Submission
    THRESHOLD = 0.7093
    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric {final_metric:.4f} > {THRESHOLD}. Generating submission..."
        )

        test_loader = get_test_loader(tokenizer)
        predictions = []
        ids = []

        # We need to match IDs. The test_loader yields batches.
        # We will read the test file to get IDs and sentiments in order.
        test_df = pd.read_csv(Config.TEST_META_PATH)

        # We iterate the loader and the dataframe simultaneously
        # The loader preserves order.

        iterator = iter(test_loader)
        row_idx = 0

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                offsets = batch["offsets"].numpy()
                texts = batch["text"]
                sentiments = batch["sentiment"]  # List of strings

                # Get model predictions for the batch
                # We only need them for non-neutrals, but we run batch for simplicity
                start_logits, end_logits = model(input_ids, attention_mask)
                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                batch_size = input_ids.size(0)

                for i in range(batch_size):
                    text_id = test_df.iloc[row_idx]["textID"]
                    sentiment = sentiments[i]
                    original_text = texts[i]
                    offset = offsets[i]

                    if sentiment == "neutral":
                        pred_text = original_text
                    else:
                        # Decode
                        start_p = start_probs[i]
                        end_p = end_probs[i]
                        score_mat = np.outer(start_p, end_p)
                        score_mat = np.triu(score_mat)
                        best_idx = np.argmax(score_mat)
                        best_start_idx, best_end_idx = np.unravel_index(
                            best_idx, score_mat.shape
                        )

                        if best_start_idx >= len(offset) or best_end_idx >= len(offset):
                            pred_text = original_text
                        else:
                            char_start = offset[best_start_idx][0]
                            char_end = offset[best_end_idx][1]
                            if char_start == 0 and char_end == 0:
                                pred_text = original_text
                            else:
                                pred_text = original_text[char_start:char_end]

                    # Ensure quoted string for CSV format requirements
                    predictions.append(f'"{pred_text}"')
                    ids.append(text_id)

                    row_idx += 1

        # Save submission
        sub_df = pd.DataFrame({"textID": ids, "selected_text": predictions})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_metric:.4f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
