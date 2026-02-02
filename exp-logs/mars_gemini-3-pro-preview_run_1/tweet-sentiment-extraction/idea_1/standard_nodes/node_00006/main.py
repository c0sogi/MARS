import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.engine import run_training, run_inference
from library.dataset import process_data, TweetDataset
from library.model import TweetModel
from library.utils import jaccard


def analyze_failures(val_loader, model, device):
    """
    Performs inference on the validation set to compute the final metric
    and analyze correlations between error magnitude and input features.
    """
    model.eval()

    all_jaccards = []
    text_lens = []
    sentiments_encoded = []

    # Mapping for sentiment correlation analysis
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}

    print("Running validation inference for analysis...")

    with torch.no_grad():
        for d in val_loader:
            input_ids = d["ids"].to(device)
            attention_mask = d["mask"].to(device)
            offsets = d["offsets"].numpy()
            orig_tweet = d["orig_tweet"]
            orig_selected = d["orig_selected"]
            sentiment = d["sentiment"]

            # Feature extraction for analysis
            batch_lens = [len(str(t)) for t in orig_tweet]
            batch_sents = [sentiment_map[s] for s in sentiment]

            text_lens.extend(batch_lens)
            sentiments_encoded.extend(batch_sents)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                # Apply Neutral Heuristic
                if sentiment[i] == "neutral":
                    pred_text = orig_tweet[i]
                else:
                    idx_start = np.argmax(start_probs[i])
                    idx_end = np.argmax(end_probs[i])

                    if idx_start > idx_end:
                        idx_end = idx_start

                    cur_offsets = offsets[i]
                    # Ensure indices are within bounds
                    if idx_start >= len(cur_offsets):
                        idx_start = len(cur_offsets) - 1
                    if idx_end >= len(cur_offsets):
                        idx_end = len(cur_offsets) - 1

                    # Reconstruct text
                    text_clean = " " + " ".join(orig_tweet[i].split())
                    char_start = cur_offsets[idx_start][0]
                    char_end = cur_offsets[idx_end][1]

                    if char_start < len(text_clean) and char_end <= len(text_clean):
                        pred_text = text_clean[char_start:char_end]
                    else:
                        pred_text = orig_tweet[i]

                    pred_text = pred_text.strip()

                score = jaccard(pred_text, orig_selected[i])
                all_jaccards.append(score)

    # 1. Final Metric
    final_metric = np.mean(all_jaccards)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    # Error Magnitude = 1 - Jaccard Score
    errors = 1.0 - np.array(all_jaccards)

    # Correlation with Text Length
    if len(errors) > 1:
        corr_len = np.corrcoef(errors, text_lens)[0, 1]
        print(f"Correlation between Error Magnitude and Text Length: {corr_len}")

        # Correlation with Sentiment
        corr_sent = np.corrcoef(errors, sentiments_encoded)[0, 1]
        print(f"Correlation between Error Magnitude and Sentiment Class: {corr_sent}")
    else:
        print("Not enough samples for correlation analysis.")

    return final_metric


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 1. Training
    # ---------------------------------------------------------
    # Executes the training loop defined in library.engine
    # This filters neutrals, trains on pos/neg, and saves the best model.
    run_training(epochs=Config.EPOCHS, debug=Config.DEBUG)

    # ---------------------------------------------------------
    # 2. Validation & Failure Analysis
    # ---------------------------------------------------------
    # We reload the validation data and the best model to perform
    # the required detailed analysis and final metric calculation.

    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load validation metadata
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Process validation data (utilizing cache if available from training step)
    val_data = process_data(
        df_val,
        tokenizer,
        Config.MAX_LEN,
        Config.WORKING_DIR,
        prefix="val",
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    val_dataset = TweetDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model saved during training
    model = TweetModel(Config)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model checkpoint not found. Analysis will use random weights.")

    model.to(device)

    # Run analysis
    final_score = analyze_failures(val_loader, model, device)

    # ---------------------------------------------------------
    # 3. Submission
    # ---------------------------------------------------------
    # Generates predictions for the test set and saves to submission.csv
    # Only if the score improves upon the previous baseline
    baseline_score = 0.6866348483059627
    if final_score > baseline_score:
        print(
            f"Validation score {final_score:.5f} exceeds baseline {baseline_score:.5f}. Generating submission..."
        )
        run_inference(debug=Config.DEBUG)
    else:
        print(
            f"Validation score {final_score:.5f} did not exceed baseline {baseline_score:.5f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
