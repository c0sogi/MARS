import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import gc
from transformers import AutoTokenizer

# Import from provided library
from library.config import Config
from library.utils import seed_everything, compute_accuracy
from library.modeling import RouterModel, GeneratorModel
from library.trainer_router import train_router
from library.trainer_generator import train_generator
from library.inference_pipeline import predict_all
from library.normalization_rules import apply_rule
from library.dataset import prepare_router_data

# Configure logging to suppress verbose output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_validation_inference():
    """
    Runs the full inference pipeline on the validation set to compute the
    actual normalization accuracy and perform failure analysis.
    """
    logger.info("Starting Validation Inference...")
    device = torch.device(Config.DEVICE)

    # 1. Load Validation Data
    # We load the raw CSV to get the ground truth 'after' and 'before'
    val_df = pd.read_csv(Config.VAL_FILE, keep_default_na=False)

    # Group for Router Inference (Sentence level)
    # We need to reconstruct the grouped format expected by the model
    val_grouped = (
        val_df.groupby("sentence_id", sort=False)
        .agg({"before": list, "id": list, "after": list, "class": list})
        .reset_index()
    )
    val_grouped.rename(columns={"before": "tokens", "id": "token_ids"}, inplace=True)

    # 2. Router Inference
    router_path = os.path.join(Config.CHECKPOINT_DIR, "router_best")
    if not os.path.exists(router_path):
        logger.warning("Router checkpoint not found. Using base model.")
        router_path = Config.ROUTER_MODEL_NAME

    router_model = RouterModel.from_pretrained(router_path).to(device)
    router_tokenizer = AutoTokenizer.from_pretrained(router_path)
    router_model.eval()

    all_token_ids = []
    all_pred_labels = []
    all_raw_tokens = []
    all_sentence_ids = []
    all_ground_truth = []

    batch_size = Config.ROUTER_BATCH_SIZE * 2

    # Process in batches
    for i in range(0, len(val_grouped), batch_size):
        batch_df = val_grouped.iloc[i : i + batch_size]
        # Cite debug_lesson_9: Ensure inner elements are lists, not numpy arrays
        batch_tokens = [
            list(t) if isinstance(t, np.ndarray) else t
            for t in batch_df["tokens"].tolist()
        ]
        batch_ids = batch_df["token_ids"].tolist()
        batch_s_ids = batch_df["sentence_id"].tolist()
        batch_gt = batch_df["after"].tolist()

        encodings = router_tokenizer(
            batch_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=Config.ROUTER_MAX_LEN,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = router_model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()

        for idx, (pred_seq, token_list, id_list, s_id, gt_list) in enumerate(
            zip(preds, batch_tokens, batch_ids, batch_s_ids, batch_gt)
        ):
            word_ids = encodings.word_ids(batch_index=idx)
            aligned_labels = []
            prev_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    continue
                if word_idx != prev_word_idx:
                    label_id = pred_seq[
                        (
                            len(aligned_labels)
                            if len(aligned_labels) < len(pred_seq)
                            else -1
                        )
                    ]  # Safety index
                    # Actually we need to index into pred_seq using the loop index j corresponding to word_ids
                    # But word_ids is flattened. Let's use enumerate on word_ids
                    pass

            # Re-loop correctly
            aligned_labels = []
            prev_word_idx = None
            for j, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != prev_word_idx:
                    label_id = pred_seq[j]
                    label_str = Config.ID2LABEL.get(label_id, "PLAIN")
                    aligned_labels.append(label_str)
                    prev_word_idx = word_idx

            # Truncate or Pad
            if len(aligned_labels) < len(token_list):
                aligned_labels.extend(
                    ["PLAIN"] * (len(token_list) - len(aligned_labels))
                )
            elif len(aligned_labels) > len(token_list):
                aligned_labels = aligned_labels[: len(token_list)]

            all_token_ids.extend(id_list)
            all_pred_labels.extend(aligned_labels)
            all_raw_tokens.extend(token_list)
            all_sentence_ids.extend([s_id] * len(token_list))
            all_ground_truth.extend(gt_list)

    pred_df = pd.DataFrame(
        {
            "id": all_token_ids,
            "token": all_raw_tokens,
            "class": all_pred_labels,
            "sentence_id": all_sentence_ids,
            "ground_truth": all_ground_truth,
        }
    )

    # Initialize prediction with raw token
    pred_df["predicted"] = pred_df["token"]

    # 3. Path A: Rules
    path_a_mask = pred_df["class"].isin(Config.PATH_A_CLASSES)
    if path_a_mask.any():
        pred_df.loc[path_a_mask, "predicted"] = pred_df[path_a_mask].apply(
            lambda row: apply_rule(row["token"], row["class"]), axis=1
        )

    # 4. Path B: Generator
    path_b_mask = pred_df["class"].isin(Config.PATH_B_CLASSES)
    if path_b_mask.any():
        generator_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best")
        if os.path.exists(generator_path):
            gen_model = GeneratorModel.from_pretrained(generator_path).to(device)
            gen_tokenizer = AutoTokenizer.from_pretrained(generator_path)
            gen_model.eval()

            sentence_map = pred_df.groupby("sentence_id")["token"].apply(list).to_dict()
            path_b_indices = pred_df[path_b_mask].index
            input_texts = []

            for idx in path_b_indices:
                row = pred_df.loc[idx]
                s_id = row["sentence_id"]
                # Extract token index from ID (e.g. "10_5" -> 5)
                try:
                    t_id = int(str(row["id"]).split("_")[-1])
                except:
                    t_id = 0

                ctx_tokens = sentence_map.get(s_id, [])
                label = row["class"]

                start = max(0, t_id - Config.CONTEXT_WINDOW_SIZE)
                end = min(len(ctx_tokens), t_id + Config.CONTEXT_WINDOW_SIZE + 1)

                left = " ".join(ctx_tokens[start:t_id])
                target = ctx_tokens[t_id] if t_id < len(ctx_tokens) else row["token"]
                right = " ".join(ctx_tokens[t_id + 1 : end])

                input_str = f"{label} {left} {Config.TARGET_START_TOKEN} {target} {Config.TARGET_END_TOKEN} {right}"
                input_texts.append(input_str)

            # Generate
            gen_batch_size = Config.GENERATOR_BATCH_SIZE * 4
            generated_outputs = []

            for i in range(0, len(input_texts), gen_batch_size):
                batch_texts = input_texts[i : i + gen_batch_size]
                inputs = gen_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=Config.GENERATOR_MAX_INPUT_LEN,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = gen_model.generate(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        max_length=Config.GENERATOR_MAX_TARGET_LEN,
                    )
                decoded = gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_outputs.extend(decoded)

            pred_df.loc[path_b_indices, "predicted"] = generated_outputs

    # 5. Compute Metric
    # Exact match required
    correct_mask = pred_df["predicted"] == pred_df["ground_truth"]
    accuracy = correct_mask.mean()

    print(f"Final Validation Metric: {accuracy}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")
    pred_df["is_error"] = (~correct_mask).astype(int)
    pred_df["token_len"] = pred_df["token"].str.len()

    # Calculate sentence length map
    sent_lens = pred_df.groupby("sentence_id").size()
    pred_df["sent_len"] = pred_df["sentence_id"].map(sent_lens)

    # Correlation
    corr_token_len = pred_df["token_len"].corr(pred_df["is_error"])
    corr_sent_len = pred_df["sent_len"].corr(pred_df["is_error"])

    print("-" * 30)
    print("Failure Analysis Correlations (Error vs Feature):")
    print(f"Token Length: {corr_token_len:.4f}")
    print(f"Sentence Length: {corr_sent_len:.4f}")
    print("-" * 30)

    return accuracy


def main():
    seed_everything(Config.SEED)

    # ==========================================
    # 0. Fast Baseline Configuration
    # ==========================================
    # Override Config for speed within 2 hours
    Config.ROUTER_EPOCHS = 1
    Config.GENERATOR_EPOCHS = 1

    # ==========================================
    # 1. Train Router
    # ==========================================
    logger.info("Step 1: Training Router...")
    # We rely on the hard-negative mining in prepare_router_data to keep dataset size manageable
    train_router(epochs=Config.ROUTER_EPOCHS)

    # ==========================================
    # 2. Train Generator
    # ==========================================
    logger.info("Step 2: Training Generator...")
    # Generator only trains on Path B classes (<5% of data), so it is naturally fast
    train_generator(epochs=Config.GENERATOR_EPOCHS)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    logger.info("Step 3: Validation and Analysis...")
    val_metric = run_validation_inference()

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.9906485140019942

    if val_metric > THRESHOLD:
        logger.info(
            f"Validation metric ({val_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_all(load_cached_data=True)
    else:
        logger.info(
            f"Validation metric ({val_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
