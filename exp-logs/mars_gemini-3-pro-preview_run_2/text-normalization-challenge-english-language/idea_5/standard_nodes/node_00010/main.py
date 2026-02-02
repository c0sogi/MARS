import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import load_data, get_tokenizer, TextNormalizationDataset
from library.model import TransformerCRF
from library.engine import fit, predict
from library.normalization_rules import normalize_token


def run():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    logger = get_logger()

    # Ensure output directory exists for the final submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    logger.info("Initializing pipeline...")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Load grouped dataframes. load_data handles caching and strategic sampling.
    logger.info("Loading datasets...")
    train_df = load_data("train", load_cached_data=True)
    val_df = load_data("val", load_cached_data=True)

    # Initialize Tokenizer
    tokenizer = get_tokenizer()

    # Create PyTorch Datasets
    # We map labels to IDs using Config.LABEL2ID
    train_dataset = TextNormalizationDataset(
        train_df, tokenizer, Config.LABEL2ID, is_test=False
    )
    val_dataset = TextNormalizationDataset(
        val_df, tokenizer, Config.LABEL2ID, is_test=False
    )

    logger.info(f"Train dataset size: {len(train_dataset)} sentences")
    logger.info(f"Val dataset size: {len(val_dataset)} sentences")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    logger.info("Initializing model...")
    model = TransformerCRF()
    model.to(Config.DEVICE)

    # ==========================================
    # 4. Training
    # ==========================================
    # fit() handles the training loop, validation loss monitoring, and checkpointing
    logger.info("Starting training...")
    fit(model, train_dataset, val_dataset)

    # ==========================================
    # 5. Validation & Metric Calculation
    # ==========================================
    logger.info("Starting full validation inference for metric calculation...")

    # Load raw validation metadata to get ground truth 'after' texts for exact string matching
    # The grouped val_df used for the Dataset object aggregates tokens but doesn't store the 'after' column conveniently
    val_meta_df = pd.read_csv(Config.VAL_META_PATH, keep_default_na=False)
    truth_map = dict(zip(val_meta_df["id"], val_meta_df["after"]))
    class_map = dict(
        zip(val_meta_df["id"], val_meta_df["class"])
    )  # For failure analysis

    # Prepare for inference
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    total_tokens = 0
    correct_tokens = 0

    # For Failure Analysis
    analysis_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)

            # Forward pass to get tag sequences (List[List[int]])
            batch_tag_seqs = model(input_ids, attention_mask)

            # Process each sentence in the batch
            for i, tag_seq in enumerate(batch_tag_seqs):
                global_idx = batch_idx * val_loader.batch_size + i
                if global_idx >= len(val_dataset):
                    break

                # Get raw info from the dataset
                raw_tokens = val_dataset.tokens_list[global_idx]
                row_ids = val_dataset.ids_list[global_idx]

                # Re-tokenize for alignment (subword -> word)
                encoding = tokenizer(
                    raw_tokens,
                    is_split_into_words=True,
                    max_length=Config.MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                word_ids = encoding.word_ids()

                # Align predicted tags to words
                pred_labels = {}
                current_word_idx = -1
                limit = min(len(tag_seq), len(word_ids))

                for t in range(limit):
                    w_id = word_ids[t]
                    tag_id = tag_seq[t]
                    if w_id is None:
                        continue
                    if w_id != current_word_idx:
                        label_str = Config.ID2LABEL.get(tag_id, "PLAIN")
                        pred_labels[w_id] = label_str
                        current_word_idx = w_id

                # Compare with ground truth
                for t_idx, token_text in enumerate(raw_tokens):
                    token_id = row_ids[t_idx]

                    # Prediction Step 1: Get Class
                    lbl = pred_labels.get(t_idx, "PLAIN")

                    # Prediction Step 2: Apply Normalization Rule
                    pred_text = normalize_token(token_text, lbl)

                    # Ground Truth
                    true_text = truth_map.get(token_id, "")

                    # Metric Update
                    is_correct = pred_text == true_text
                    if is_correct:
                        correct_tokens += 1
                    total_tokens += 1

                    # Failure Analysis Data Collection
                    # Get class index for correlation
                    true_class = class_map.get(token_id, "PLAIN")
                    class_idx = Config.LABEL2ID.get(true_class, 0)

                    analysis_records.append(
                        {
                            "len_before": len(token_text),
                            "class_idx": class_idx,
                            "is_error": 0 if is_correct else 1,
                        }
                    )

    # Calculate and print final metric
    final_metric = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    logger.info("Performing failure analysis...")
    df_analysis = pd.DataFrame(analysis_records)

    if not df_analysis.empty:
        # Correlation Analysis
        corr_len = df_analysis["len_before"].corr(df_analysis["is_error"])
        corr_class = df_analysis["class_idx"].corr(df_analysis["is_error"])

        print(f"Correlation (Token Length vs Error): {corr_len:.4f}")
        print(f"Correlation (Class ID vs Error): {corr_class:.4f}")

        # Error rate by class
        df_analysis["class_name"] = df_analysis["class_idx"].map(Config.ID2LABEL)
        class_stats = df_analysis.groupby("class_name")["is_error"].agg(
            ["mean", "count"]
        )
        # Filter classes with at least 100 samples to be statistically relevant
        class_stats = class_stats[class_stats["count"] > 100].sort_values(
            "mean", ascending=False
        )

        print("\nTop 5 Classes with Highest Error Rates:")
        print(class_stats.head(5))
    else:
        logger.warning("No analysis data available.")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    THRESHOLD = 0.973229717044087

    if final_metric > THRESHOLD:
        logger.info(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")

        # Load test data
        test_df = load_data("test", load_cached_data=True)
        test_dataset = TextNormalizationDataset(
            test_df, tokenizer, Config.LABEL2ID, is_test=True
        )

        # Run prediction
        # engine.predict saves to Config.SUBMISSION_PATH (in ./working)
        predict(model, test_dataset)

        # Move/Copy to final location ./submission/submission.csv
        if os.path.exists(Config.SUBMISSION_PATH):
            shutil.copy(Config.SUBMISSION_PATH, FINAL_SUBMISSION_PATH)
            logger.info(f"Final submission saved to {FINAL_SUBMISSION_PATH}")
        else:
            logger.error(f"Submission file not found at {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"Metric {final_metric} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    run()
