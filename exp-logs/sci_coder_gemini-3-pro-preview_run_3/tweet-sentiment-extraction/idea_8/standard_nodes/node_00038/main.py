import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Suppress progress bars
os.environ["TQDM_DISABLE"] = "1"

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.engine import run_training, generate_submission
from library.data import process_data, TweetDataset
from library.model import TweetModel


def run():
    # =========================================================================
    # 1. Configuration Override for Fast Baseline
    # =========================================================================
    # We modify the Config class attributes directly to control the execution
    Config.EPOCHS = 1
    Config.N_FOLDS = 1  # Train only Fold 0 to ensure completion within time limit
    Config.TRAIN_BATCH_SIZE = 16
    Config.VALID_BATCH_SIZE = 32
    # Enable AWP immediately since we only run 1 epoch
    Config.AWP_START_EPOCH = 0

    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("Starting Training (Fast Baseline)...")
    # This will train Fold 0 and save 'model_fold_0.bin'
    run_training()

    # =========================================================================
    # 3. Evaluation on Hold-out Set
    # =========================================================================
    print("Starting Evaluation on Hold-out Set...")

    # Load hold-out validation metadata
    val_df = pd.read_csv(Config.VAL_META_PATH)

    # Preprocess validation data
    # We use 'is_test=True' to generate input_ids/masks/offsets without requiring target labels in the dataset class
    # (We handle metric calculation manually)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    val_data = process_data(
        val_df,
        tokenizer,
        Config.MAX_LEN,
        is_test=True,
        cache_prefix="val_holdout",
        load_cached_data=False,
    )

    val_dataset = TweetDataset(
        input_ids=val_data["input_ids"],
        attention_mask=val_data["attention_mask"],
        orig_texts=val_data["orig_texts"],
        offsets=val_data["offsets"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the trained model (Fold 0)
    model = TweetModel()
    model.to(Config.DEVICE)
    model_path = os.path.join(Config.WORKING_DIR, "model_fold_0.bin")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    # Inference Loop
    pred_strs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            offsets = batch["offsets"].cpu().numpy()
            orig_texts = batch["orig_text"]

            start_logits, end_logits = model(input_ids, attention_mask)
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(input_ids)):
                text = orig_texts[i]
                s_p = start_probs[i]
                e_p = end_probs[i]

                # Summation Decoding: Maximize P_start[s] + P_end[e]
                score_mat = np.expand_dims(s_p, 1) + np.expand_dims(e_p, 0)
                seq_len = len(s_p)
                mask = np.triu(np.ones((seq_len, seq_len)))
                score_mat = score_mat * mask + (1 - mask) * -1e10

                best_idx = np.argmax(score_mat)
                start_idx, end_idx = np.unravel_index(best_idx, score_mat.shape)

                current_offsets = offsets[i]
                if start_idx < len(current_offsets) and end_idx < len(current_offsets):
                    char_start = current_offsets[start_idx][0]
                    char_end = current_offsets[end_idx][1]
                    pred_text = text[char_start:char_end]
                else:
                    pred_text = text

                pred_strs.append(pred_text)

    # Apply Neutral Rule: If sentiment is neutral, prediction is the full text
    val_df["pred_selected_text"] = pred_strs
    val_df.loc[val_df["sentiment"] == "neutral", "pred_selected_text"] = val_df.loc[
        val_df["sentiment"] == "neutral", "text"
    ]

    # Calculate Jaccard Scores
    scores = []
    for idx, row in val_df.iterrows():
        scores.append(jaccard(row["selected_text"], row["pred_selected_text"]))

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("Performing Failure Analysis...")
    val_df["jaccard"] = scores
    val_df["error"] = 1.0 - val_df["jaccard"]
    val_df["text_len"] = val_df["text"].apply(lambda x: len(str(x)))

    # Correlation between Error Magnitude (1-Jaccard) and Input Length
    corr_len, _ = pearsonr(val_df["error"], val_df["text_len"])
    print(f"Correlation between Error and Text Length: {corr_len}")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    THRESHOLD = 0.7164761348654044
    if final_metric > THRESHOLD:
        # generate_submission uses Config.N_FOLDS to load models.
        # Since we set N_FOLDS=1, it will correctly use only model_fold_0.bin
        generate_submission()
    else:
        print(
            f"Metric {final_metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
