import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add current directory to sys.path to ensure imports work
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, normalize_text, jaccard
from library.dataset import get_data, TweetDataset
from library.model import TweetModel
from library.engine import train_fn

# Suppress warnings
warnings.filterwarnings("ignore")


def predict_fn(data_loader, model, device, df, is_test=False):
    """
    Generates predictions using Joint Logit Decoding.
    """
    model.eval()
    start_preds = []
    end_preds = []

    # Inference loop to collect logits
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_preds.append(start_logits.cpu().numpy())
            end_preds.append(end_logits.cpu().numpy())

    start_preds = np.concatenate(start_preds)
    end_preds = np.concatenate(end_preds)

    predictions = []
    dataset_offsets = data_loader.dataset.offsets

    # Decoding loop
    for i in range(len(df)):
        row = df.iloc[i]
        # Apply Normalize-First strategy
        text = normalize_text(row["text"])
        offsets = dataset_offsets[i]

        s_logits = start_preds[i]
        e_logits = end_preds[i]

        # Joint Logit Decoding: Maximize sum of start and end logits
        sum_logits = s_logits[:, None] + e_logits[None, :]

        # Enforce start <= end constraint
        mask = np.triu(np.ones_like(sum_logits))
        sum_logits = np.where(mask == 1, sum_logits, -np.inf)

        best_idx = np.argmax(sum_logits)
        best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

        # Extract substring using offsets
        if best_start < len(offsets) and best_end < len(offsets):
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            pred_text = text[start_char:end_char]
        else:
            pred_text = text

        predictions.append(pred_text)

    return predictions


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load Metadata
    df_train_full = pd.read_csv(Config.TRAIN_META)
    df_val_full = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    # Filter out 'neutral' tweets for model training (Identity mapping rule applies)
    df_train = df_train_full[df_train_full["sentiment"] != "neutral"].reset_index(
        drop=True
    )
    df_val_filtered = df_val_full[df_val_full["sentiment"] != "neutral"].reset_index(
        drop=True
    )

    # Get processed data (utilizing cache)
    train_data = get_data(
        df_train, tokenizer, Config, cache_name="train_pos_neg", load_cached_data=True
    )
    val_data = get_data(
        df_val_filtered,
        tokenizer,
        Config,
        cache_name="val_pos_neg",
        load_cached_data=True,
    )

    train_dataset = TweetDataset(train_data, Config)
    val_dataset = TweetDataset(val_data, Config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = TweetModel(Config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        _ = train_fn(train_loader, model, optimizer, device, scheduler)

    # 5. Validation Assessment
    # Predict on the positive/negative subset
    val_preds_filtered = predict_fn(val_loader, model, device, df_val_filtered)

    # Integrate predictions into the full validation set
    df_val_full["prediction"] = ""

    # Apply deterministic rule for neutrals
    neutral_mask = df_val_full["sentiment"] == "neutral"
    df_val_full.loc[neutral_mask, "prediction"] = df_val_full.loc[neutral_mask, "text"]

    # Fill model predictions for non-neutrals
    df_val_full.loc[~neutral_mask, "prediction"] = val_preds_filtered

    # Compute Jaccard Score
    df_val_full["jaccard"] = df_val_full.apply(
        lambda x: jaccard(x["prediction"], x["selected_text"]), axis=1
    )
    final_metric = df_val_full["jaccard"].mean()

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    df_val_full["error"] = 1.0 - df_val_full["jaccard"]

    # Feature Engineering for Analysis
    df_val_full["text_len"] = df_val_full["text"].astype(str).apply(len)
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df_val_full["sentiment_enc"] = df_val_full["sentiment"].map(sentiment_map)

    # Calculate Correlations
    corr_len = df_val_full["error"].corr(df_val_full["text_len"])
    corr_sent = df_val_full["error"].corr(df_val_full["sentiment_enc"])

    print("Failure Analysis Correlations:")
    print(f"Error vs Text Length: {corr_len}")
    print(f"Error vs Sentiment: {corr_sent}")

    # 7. Submission Generation
    THRESHOLD = 0.7043342108129372

    if final_metric > THRESHOLD:
        # Prepare test dataframe
        df_test["selected_text"] = ""

        # Handle neutrals
        test_neutral_mask = df_test["sentiment"] == "neutral"
        df_test.loc[test_neutral_mask, "selected_text"] = df_test.loc[
            test_neutral_mask, "text"
        ]

        # Handle non-neutrals
        df_test_filtered = df_test[~test_neutral_mask].reset_index(drop=True)

        if len(df_test_filtered) > 0:
            test_data = get_data(
                df_test_filtered,
                tokenizer,
                Config,
                cache_name="test_pos_neg",
                load_cached_data=True,
            )
            test_dataset = TweetDataset(test_data, Config, is_test=True)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=Config.VALID_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            test_preds = predict_fn(
                test_loader, model, device, df_test_filtered, is_test=True
            )
            df_test.loc[~test_neutral_mask, "selected_text"] = test_preds

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        # quoting=1 corresponds to csv.QUOTE_ALL
        df_test[["textID", "selected_text"]].to_csv(
            submission_path, index=False, quoting=1
        )


if __name__ == "__main__":
    main()
