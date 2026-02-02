import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library import train, inference, dataset, model, engine, utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    utils.seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We limit to 1 fold and 2 epochs to meet the time constraint while ensuring convergence.
    # N_FOLDS must be >= 2 for StratifiedKFold. We set it to 5 to keep the 80/20 split.
    Config.N_FOLDS = 5
    Config.EPOCHS = 2
    Config.TRAIN_BATCH_SIZE = 16  # Increased batch size for A100 efficiency
    Config.VALID_BATCH_SIZE = 32

    # Ensure necessary directories exist
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(
        f"Configuration: Folds={Config.N_FOLDS}, Epochs={Config.EPOCHS}, Device={Config.DEVICE}"
    )

    # 2. Train Model
    # We train only Fold 0 as a representative baseline.
    print("\n--- Starting Training (Fold 0) ---")
    train.run_fold(0)
    print("Training of Fold 0 complete.")

    # 3. Validation on Hold-out Set
    # We use the explicit hold-out validation set from metadata/val.csv
    print("\n--- Starting Validation on Hold-out Set ---")

    val_df = pd.read_csv(Config.VAL_FILE)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process validation data
    # We use a distinct cache file for this hold-out set
    cache_path = os.path.join(
        Config.CACHE_DIR, f"cached_holdout_val_{Config.MAX_LEN}.npz"
    )
    data = dataset.process_data(
        val_df,
        tokenizer,
        Config.MAX_LEN,
        cache_path,
        load_cached_data=True,
        is_test=False,
    )

    # Create Validation Dataset and Loader
    val_dataset = dataset.TweetDataset(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        token_type_ids=data["token_type_ids"],
        offsets=data["offsets"],
        original_texts=val_df["text"].values,
        sentiments=val_df["sentiment"].values,
        start_tokens=data["start_tokens"],
        end_tokens=data["end_tokens"],
        selected_texts=val_df["selected_text"].values,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the trained model
    device = Config.DEVICE
    model_instance = model.TweetModel(Config)
    model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "model_fold_0.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.to(device)
    model_instance.eval()

    # Run Inference on Validation Set
    start_preds, end_preds, _ = engine.eval_fn(val_loader, model_instance, device)

    # Decode Predictions and Compute Jaccard
    jaccard_scores = []
    errors = []
    text_lens = []
    sentiment_codes = []  # 0: neutral, 1: positive, 2: negative

    texts = val_df["text"].values
    sentiments = val_df["sentiment"].values
    selected_texts = val_df["selected_text"].values
    offsets = data["offsets"]

    # Pre-compute mask for decoding
    ones = np.ones((Config.MAX_LEN, Config.MAX_LEN))
    triu_mask = np.triu(ones, k=0)

    for i in range(len(val_df)):
        text = str(texts[i])
        sentiment = str(sentiments[i])
        selected_text = str(selected_texts[i])
        offset = offsets[i]

        start_logits = start_preds[i]
        end_logits = end_preds[i]

        # Apply Neutral Heuristic
        if sentiment == "neutral" and Config.NEUTRAL_FULL_TEXT:
            pred_text = text
        else:
            # Decode span
            sum_matrix = start_logits[:, None] + end_logits[None, :]
            sum_matrix = sum_matrix * triu_mask + (1 - triu_mask) * -1e9
            flat_idx = np.argmax(sum_matrix)
            start_idx = flat_idx // Config.MAX_LEN
            end_idx = flat_idx % Config.MAX_LEN
            pred_text = utils.get_selected_text(text, start_idx, end_idx, offset)

        # Compute Score
        score = utils.jaccard(pred_text, selected_text)
        jaccard_scores.append(score)

        # Collect data for failure analysis
        errors.append(1.0 - score)
        text_lens.append(len(text))

        if sentiment == "neutral":
            sentiment_codes.append(0)
        elif sentiment == "positive":
            sentiment_codes.append(1)
        else:
            sentiment_codes.append(2)

    final_metric = np.mean(jaccard_scores)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    fa_df = pd.DataFrame(
        {"error": errors, "text_len": text_lens, "sentiment_code": sentiment_codes}
    )

    correlations = fa_df.corr()["error"]
    print("Correlation between Error Magnitude (1-Jaccard) and Input Features:")
    print(correlations)

    # 5. Submission Generation
    THRESHOLD = 0.7092567967346735

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # Run inference on test set
        # Config.N_FOLDS is 1, so inference.predict will correctly use only model_fold_0
        inference.predict()
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
