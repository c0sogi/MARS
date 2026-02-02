import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Import from provided library files
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import BiLSTMTagger
from library.trainer import Trainer
from library.inference import generate_submission, normalize_text
from library.utils import get_logger

# Initialize logger
logger = get_logger("runfile")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for optimized execution
    # We use the full dataset to maximize Knowledge Base coverage and Classifier performance.
    # Cite solution_lesson_node_00005: Decoupling disambiguation from deterministic transformation.
    Config.MAX_TRAIN_SAMPLES = None
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 1024  # Increased for A100 efficiency
    Config.LEARNING_RATE = 3e-4
    Config.USE_CLASS_WEIGHTS = False
    Config.NUM_WORKERS = 8

    # Clean up old artifacts to prevent loading stale models or data (Fail Fast logic)
    # Cite debug_lesson_1
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    logger.info("Configuration set for fast baseline.")
    logger.info(f"Max Train Samples: {Config.MAX_TRAIN_SAMPLES}")
    logger.info(f"Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading DataLoaders and Resources...")
    # This handles preprocessing, caching, and loader creation
    train_loader, val_loader, test_loader, vocab, kb = get_dataloaders(
        load_cached_data=True
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    logger.info("Initializing Bi-LSTM Model...")
    model = BiLSTMTagger(
        vocab_size=len(vocab.token2id),
        num_classes=len(vocab.class2id),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL,
    )

    # ==========================================
    # 4. Training
    # ==========================================
    logger.info("Starting Training Process...")
    trainer = Trainer(model, train_loader, val_loader, vocab)
    trainer.fit()

    # ==========================================
    # 5. Validation & Metric Calculation
    # ==========================================
    logger.info("Performing Final Validation Evaluation...")

    # The Trainer computes class-level accuracy. We must compute the task-specific
    # metric: Exact String Match of the normalized text.

    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    # Load the raw validation data to get the ground truth strings.
    # We must replicate the subsampling logic from get_dataloaders to ensure alignment
    # with the val_loader.
    val_grouped_path = os.path.join(Config.WORKING_DIR, "val_grouped.parquet")
    df_val = pd.read_parquet(val_grouped_path)

    if Config.MAX_TRAIN_SAMPLES:
        limit = Config.MAX_TRAIN_SAMPLES
        # Logic matches get_dataloaders: df_val = df_val.head(limit // 5)
        df_val = df_val.head(limit // 5)

    val_records = df_val.to_dict("records")

    all_preds = []
    all_targets = []

    # Data for failure analysis
    analysis_records = []

    current_idx = 0
    total_val_sents = len(val_records)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            seq_len = batch["seq_len"]
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            logits = model(input_ids, seq_len, attention_mask)
            # Get class predictions
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

            batch_size = input_ids.size(0)

            for i in range(batch_size):
                if current_idx >= total_val_sents:
                    break

                record = val_records[current_idx]
                current_idx += 1

                raw_tokens = record["before"]
                target_tokens = record["after"]

                # Get predictions for this sentence
                sentence_preds = pred_ids[i]

                for t_idx, (token, target) in enumerate(zip(raw_tokens, target_tokens)):
                    # Decode class
                    if t_idx < Config.MAX_LEN:
                        class_id = sentence_preds[t_idx]
                        pred_class = vocab.id2class.get(class_id, "PLAIN")
                    else:
                        pred_class = "PLAIN"

                    # Normalize text using the KB and predicted class
                    normalized_text = normalize_text(token, pred_class, kb)

                    all_preds.append(normalized_text)
                    all_targets.append(target)

                    # Collect data for failure analysis
                    is_error = 1 if normalized_text != target else 0
                    analysis_records.append(
                        {
                            "token_len": len(token),
                            "is_error": is_error,
                            "pred_class_id": class_id if t_idx < Config.MAX_LEN else 0,
                        }
                    )

    # Compute Metric (Exact String Match)
    final_acc = accuracy_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_acc}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")
    if analysis_records:
        df_analysis = pd.DataFrame(analysis_records)

        # Correlation: Error vs Token Length
        corr_len = df_analysis["token_len"].corr(df_analysis["is_error"])

        # Correlation: Error vs Class ID (proxy for class type distribution)
        corr_class = df_analysis["pred_class_id"].corr(df_analysis["is_error"])

        print("Correlation Analysis (Error vs Features):")
        print(f"Correlation (Token Length vs Error): {corr_len}")
        print(f"Correlation (Class ID vs Error): {corr_class}")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    logger.info("Generating Submission for Test Set...")
    # generate_submission handles loading test data, model, and saving file
    generate_submission(debug=False)
    logger.info("Runfile execution complete.")


if __name__ == "__main__":
    main()
