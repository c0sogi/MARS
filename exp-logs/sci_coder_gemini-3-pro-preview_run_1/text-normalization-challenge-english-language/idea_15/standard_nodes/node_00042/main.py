import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config, set_seed
from library.utils import get_logger
from library.vocab_manager import build_vocabs
from library.feature_engineering import FeatureEngineer
from library.knowledge_base import KnowledgeBase
from library.datasets import TaggerDataset, Seq2SeqDataset
from library.trainer import ModelTrainer
from library.inference import Predictor, Seq2SeqInferenceDataset

# ------------------------------------------------------------------------------
# 1. Setup & Configuration
# ------------------------------------------------------------------------------
Config.setup()
set_seed(Config.SEED)
logger = get_logger("runfile")

# Override Config for Fast Baseline Execution
# We use a subset of data for training to meet the time limit,
# but we rely on the full Knowledge Base for high accuracy.
Config.DEBUG = True
Config.DEBUG_SIZE = 300000  # Sufficient for Tagger convergence
Config.TAGGER_EPOCHS = 1
Config.SEQ2SEQ_EPOCHS = 1
Config.BATCH_SIZE = 1024  # Maximize A100 utilization
Config.NUM_WORKERS = 8  # Speed up data loading

logger.info("Configuration configured for fast baseline run.")

# ------------------------------------------------------------------------------
# 2. Resource Initialization (Vocabs, KB, Features)
# ------------------------------------------------------------------------------
logger.info("Initializing resources...")

# Build/Load Vocabularies
word_vocab, char_vocab, class_vocab, bpe_tokenizer = build_vocabs(load_cached_data=True)

# Initialize Feature Engineer and Load Priors
fe = FeatureEngineer()
priors_df = fe.build_or_load_priors(class_vocab, load_cached_data=True)

# Build/Load Knowledge Base (Deterministic Memory)
# Note: This loads from the full training set regardless of Config.DEBUG, ensuring high performance.
kb = KnowledgeBase()
kb.build(load_cached_data=True)

# ------------------------------------------------------------------------------
# 3. Data Loading & Training
# ------------------------------------------------------------------------------
logger.info("Preparing training datasets...")

# Tagger Datasets (Debug=True for speed)
train_tagger_ds = TaggerDataset(
    Config.TRAIN_FILE,
    word_vocab,
    char_vocab,
    class_vocab,
    bpe_tokenizer,
    fe,
    priors_df,
    split="train",
    load_cached_data=True,
    debug=True,
)
val_tagger_ds_monitor = TaggerDataset(
    Config.VAL_FILE,
    word_vocab,
    char_vocab,
    class_vocab,
    bpe_tokenizer,
    fe,
    priors_df,
    split="val",
    load_cached_data=True,
    debug=True,
)

# Seq2Seq Datasets (Debug=True for speed)
train_seq2seq_ds = Seq2SeqDataset(
    Config.TRAIN_FILE,
    char_vocab,
    class_vocab,
    split="train",
    load_cached_data=True,
    debug=True,
)
val_seq2seq_ds_monitor = Seq2SeqDataset(
    Config.VAL_FILE,
    char_vocab,
    class_vocab,
    split="val",
    load_cached_data=True,
    debug=True,
)

# Initialize Trainer
trainer = ModelTrainer(device=Config.DEVICE)

# Train Tagger
logger.info("Starting Tagger Training...")
tagger_model = trainer.train_tagger(
    train_tagger_ds,
    val_tagger_ds_monitor,
    word_vocab,
    bpe_tokenizer,
    char_vocab,
    class_vocab,
)

# Train Seq2Seq
logger.info("Starting Seq2Seq Training...")
seq2seq_model = trainer.train_seq2seq(
    train_seq2seq_ds, val_seq2seq_ds_monitor, char_vocab, class_vocab
)

# ------------------------------------------------------------------------------
# 4. Full Validation (Hybrid System)
# ------------------------------------------------------------------------------
logger.info("Running Full Hybrid Validation on entire validation set...")

# Load Full Validation Dataset (Debug=False)
# This ensures the metric is calculated on the complete hold-out set as required.
val_tagger_ds_full = TaggerDataset(
    Config.VAL_FILE,
    word_vocab,
    char_vocab,
    class_vocab,
    bpe_tokenizer,
    fe,
    priors_df,
    split="val",
    load_cached_data=True,
    debug=False,
)

val_loader = DataLoader(
    val_tagger_ds_full,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
)

# Step 4.1: Tagger Inference
tagger_model.eval()
all_pred_classes = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Val Tagging"):
        word_ids = batch["word_ids"].to(Config.DEVICE).unsqueeze(1)
        bpe_ids = batch["bpe_ids"].to(Config.DEVICE).unsqueeze(1)
        char_ids = batch["char_ids"].to(Config.DEVICE).unsqueeze(1)
        regex_feats = batch["regex_feats"].to(Config.DEVICE).unsqueeze(1)
        prior_feats = batch["prior_feats"].to(Config.DEVICE).unsqueeze(1)

        logits = tagger_model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)
        logits = logits.squeeze(1)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_pred_classes.extend(preds)

# Step 4.2: Hybrid Logic (KB + Fallback)
df_val = pd.read_csv(Config.VAL_FILE, dtype=str, keep_default_na=False)
tokens = df_val["before"].astype(str).tolist()
targets = df_val["after"].astype(str).tolist()

final_predictions = [""] * len(tokens)
fallback_indices = []
fallback_inputs = []

for i, (token, pred_cls_idx) in enumerate(zip(tokens, all_pred_classes)):
    pred_cls_str = class_vocab.lookup_token(pred_cls_idx)

    # A. KB Lookup
    kb_res = kb.query(token, pred_cls_str)

    if kb_res is not None:
        final_predictions[i] = kb_res
    # B. Heuristic Copy
    elif pred_cls_str in ["PLAIN", "PUNCT"]:
        final_predictions[i] = token
    # C. Prepare for Neural Fallback
    else:
        fallback_indices.append(i)
        fallback_inputs.append((token, pred_cls_idx))

# Step 4.3: Seq2Seq Inference for Fallback
if fallback_inputs:
    logger.info(
        f"Running Seq2Seq Fallback on {len(fallback_inputs)} validation tokens..."
    )
    fb_tokens, fb_classes = zip(*fallback_inputs)

    fb_ds = Seq2SeqInferenceDataset(
        fb_tokens, fb_classes, char_vocab, Config.MAX_SEQ_LEN
    )
    fb_loader = DataLoader(
        fb_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    seq2seq_model.eval()
    generated_texts = []

    with torch.no_grad():
        for src_ids, class_ids in tqdm(fb_loader, desc="Val Seq2Seq"):
            src_ids = src_ids.to(Config.DEVICE)
            class_ids = class_ids.to(Config.DEVICE)

            output_ids = seq2seq_model.generate(src_ids, class_ids).cpu().numpy()

            for row in output_ids:
                chars = []
                for idx in row:
                    if idx == char_vocab["<eos>"]:
                        break
                    if idx == char_vocab["<sos>"]:
                        continue
                    if idx == char_vocab["<pad>"]:
                        continue
                    try:
                        chars.append(char_vocab.lookup_token(idx))
                    except:
                        pass
                generated_texts.append("".join(chars))

    # Merge results
    for idx, text in zip(fallback_indices, generated_texts):
        final_predictions[idx] = text

# Step 4.4: Compute Metric
correct_count = 0
errors = []

for i, (pred, target) in enumerate(zip(final_predictions, targets)):
    if pred == target:
        correct_count += 1
    else:
        errors.append(
            {"token": tokens[i], "target": target, "pred": pred, "len": len(tokens[i])}
        )

val_accuracy = correct_count / len(tokens)
print(f"Final Validation Metric: {val_accuracy}")

# ------------------------------------------------------------------------------
# 5. Failure Analysis
# ------------------------------------------------------------------------------
logger.info("Performing Failure Analysis...")

if errors:
    # Create a DataFrame for analysis
    # We need to compute correlation over the entire dataset, not just errors
    analysis_df = pd.DataFrame(
        {
            "len": [len(t) for t in tokens],
            "is_error": [
                1 if p != t else 0 for p, t in zip(final_predictions, targets)
            ],
        }
    )

    correlation = analysis_df["len"].corr(analysis_df["is_error"])
    print(f"Correlation between Token Length and Error: {correlation:.4f}")

    logger.info(f"Total Errors: {len(errors)}")
    if len(errors) > 0:
        logger.info(f"Sample Error: {errors[0]}")
else:
    print("Correlation between Token Length and Error: 0.0000")
    logger.info("No errors found on validation set!")

# ------------------------------------------------------------------------------
# 6. Submission
# ------------------------------------------------------------------------------
THRESHOLD = 0.9949142925818993

if val_accuracy > THRESHOLD:
    logger.info(
        f"Validation accuracy {val_accuracy} passed threshold {THRESHOLD}. Generating submission..."
    )

    # Use the Predictor class for the Test Set
    # We pass debug=False to ensure the full test set is processed
    predictor = Predictor()
    predictor.generate_submission(debug=False)

else:
    logger.info(
        f"Validation accuracy {val_accuracy} did not meet threshold {THRESHOLD}. Skipping submission."
    )
