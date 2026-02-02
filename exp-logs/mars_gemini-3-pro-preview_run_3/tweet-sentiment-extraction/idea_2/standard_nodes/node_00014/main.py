import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, jaccard, get_selected_text
from library.dataset import get_data
from library.model import TweetModel
from library.engine import fit, eval_fn


def run():
    # 1. Setup and Initialization
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # load_cached_data=True uses pre-computed .npy files for speed
    train_loader, val_loader, test_loader = get_data(load_cached_data=True)

    # 3. Model Initialization
    model = TweetModel()
    model.to(device)

    # 4. Optimizer and Scheduler Configuration
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    # The fit function handles the training epochs, validation, and saving the best model
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        save_path=Config.MODEL_SAVE_PATH,
        patience=Config.EPOCHS,  # Run for all epochs given the short duration
    )

    # 6. Final Validation Assessment
    # Load the best model saved during training
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    # Calculate final metric on the validation set
    val_loss, val_jaccard = eval_fn(val_loader, model, device)
    print(f"Final Validation Metric: {val_jaccard}")

    # 7. Failure Analysis
    # Analyze correlation between error and text length
    print("Performing Failure Analysis...")
    errors = []
    text_lengths = []

    with torch.no_grad():
        for data in val_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            texts = data["text"]
            sentiments = data["sentiment"]
            selected_texts = data.get("selected_text", [])
            offsets = data["offsets"].numpy()

            # Model inference
            start_logits, end_logits = model(input_ids, attention_mask)
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(texts)):
                orig_text = texts[i]
                target = selected_texts[i]
                sentiment = sentiments[i]
                offset = offsets[i]

                # Apply deterministic rule for neutral
                if sentiment == "neutral":
                    pred_text = orig_text
                else:
                    # Decode span for positive/negative
                    s_prob = start_probs[i]
                    e_prob = end_probs[i]
                    score_mat = np.expand_dims(s_prob, 1) + np.expand_dims(e_prob, 0)
                    score_mat = np.triu(score_mat)  # Enforce start <= end
                    best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                    pred_text = get_selected_text(
                        orig_text, best_idx[0], best_idx[1], offset
                    )

                # Calculate metrics
                score = jaccard(pred_text, target)
                errors.append(1.0 - score)
                text_lengths.append(len(orig_text))

    # Compute correlation
    if len(errors) > 1:
        corr, _ = pearsonr(errors, text_lengths)
        print(f"Correlation between Error and Text Length: {corr}")

    # 8. Submission Generation
    THRESHOLD = 0.7052319161123256

    if val_jaccard > THRESHOLD:
        print(
            f"Validation metric {val_jaccard} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        predictions = []

        with torch.no_grad():
            for data in test_loader:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                texts = data["text"]
                sentiments = data["sentiment"]
                offsets = data["offsets"].numpy()

                start_logits, end_logits = model(input_ids, attention_mask)
                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                for i in range(len(texts)):
                    orig_text = texts[i]
                    sentiment = sentiments[i]
                    offset = offsets[i]

                    if sentiment == "neutral":
                        pred_text = orig_text
                    else:
                        s_prob = start_probs[i]
                        e_prob = end_probs[i]
                        score_mat = np.expand_dims(s_prob, 1) + np.expand_dims(
                            e_prob, 0
                        )
                        score_mat = np.triu(score_mat)
                        best_idx = np.unravel_index(
                            np.argmax(score_mat), score_mat.shape
                        )
                        pred_text = get_selected_text(
                            orig_text, best_idx[0], best_idx[1], offset
                        )

                    predictions.append(pred_text)

        # Load test metadata to get textIDs
        df_test = pd.read_csv(Config.TEST_META)
        df_test["selected_text"] = predictions

        # Save submission
        # quoting=csv.QUOTE_NONNUMERIC ensures strings are quoted as required
        submission_df = df_test[["textID", "selected_text"]]
        submission_df.to_csv(
            Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"Validation metric {val_jaccard} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
