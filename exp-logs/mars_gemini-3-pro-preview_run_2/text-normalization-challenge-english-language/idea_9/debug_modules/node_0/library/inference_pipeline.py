import os
import torch
import pandas as pd
import logging
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_router_data
from library.modeling import RouterModel, GeneratorModel
from library.normalization_rules import apply_rule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def predict_all(
    load_cached_data: bool = True, debug: bool = False, debug_size: int = 1000
):
    """
    Runs the full inference pipeline on the test set.

    Steps:
    1. Runs the Router model to classify tokens.
    2. Applies deterministic rules (Path A) for rigid classes.
    3. Runs the Generator model (Path B) for ambiguous classes using context.
    4. Merges results and saves the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        debug (bool): If True, runs on a small subset of the test data.
        debug_size (int): Number of sentences to process in debug mode.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Starting Inference Pipeline...")

    # ==========================================
    # 1. Load Test Data
    # ==========================================
    logger.info("Loading test data...")
    # prepare_router_data returns a DataFrame with [sentence_id, tokens (list), token_ids (list)]
    test_grouped_df = prepare_router_data("test", load_cached_data=load_cached_data)

    if debug:
        logger.info(f"Debug mode enabled: Processing first {debug_size} sentences.")
        test_grouped_df = test_grouped_df.head(debug_size).copy()

    # ==========================================
    # 2. Router Inference (Class Prediction)
    # ==========================================
    router_path = os.path.join(Config.CHECKPOINT_DIR, "router_best")
    if not os.path.exists(router_path):
        logger.warning(
            f"Router checkpoint not found at {router_path}. Using base model {Config.ROUTER_MODEL_NAME} (Expect poor performance)."
        )
        router_path = Config.ROUTER_MODEL_NAME

    logger.info(f"Loading Router model from {router_path}...")
    router_model = RouterModel.from_pretrained(router_path).to(device)
    router_tokenizer = AutoTokenizer.from_pretrained(router_path)
    router_model.eval()

    logger.info("Running Router inference...")

    all_token_ids = []
    all_pred_labels = []
    all_raw_tokens = []
    all_sentence_ids = []

    batch_size = Config.ROUTER_BATCH_SIZE * 2
    total_samples = len(test_grouped_df)

    # Process in batches
    for i in range(0, total_samples, batch_size):
        batch_df = test_grouped_df.iloc[i : i + batch_size]
        batch_tokens = batch_df["tokens"].tolist()
        batch_ids = batch_df["token_ids"].tolist()
        batch_sentence_ids = batch_df["sentence_id"].tolist()

        # Tokenize
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
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2).cpu().numpy()

        # Align predictions to original tokens
        for idx, (pred_seq, token_list, id_list, s_id) in enumerate(
            zip(preds, batch_tokens, batch_ids, batch_sentence_ids)
        ):
            word_ids = encodings.word_ids(batch_index=idx)
            aligned_labels = []
            prev_word_idx = None

            for j, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != prev_word_idx:
                    # Start of a new token
                    label_id = pred_seq[j]
                    label_str = Config.ID2LABEL.get(label_id, "PLAIN")
                    aligned_labels.append(label_str)
                    prev_word_idx = word_idx

            # Handle potential length mismatch due to truncation
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

    # Create Prediction DataFrame
    pred_df = pd.DataFrame(
        {
            "id": all_token_ids,
            "token": all_raw_tokens,
            "class": all_pred_labels,
            "sentence_id": all_sentence_ids,
        }
    )

    # Initialize 'after' with raw tokens (copy strategy)
    pred_df["after"] = pred_df["token"]

    # ==========================================
    # 3. Path A: Deterministic Rules
    # ==========================================
    logger.info("Applying Path A (Deterministic Rules)...")
    path_a_mask = pred_df["class"].isin(Config.PATH_A_CLASSES)

    if path_a_mask.any():
        # Apply rules row-by-row
        pred_df.loc[path_a_mask, "after"] = pred_df[path_a_mask].apply(
            lambda row: apply_rule(row["token"], row["class"]), axis=1
        )

    # ==========================================
    # 4. Path B: Neural Generator
    # ==========================================
    logger.info("Applying Path B (Neural Generator)...")
    path_b_mask = pred_df["class"].isin(Config.PATH_B_CLASSES)

    if path_b_mask.any():
        generator_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best")
        if not os.path.exists(generator_path):
            logger.warning(
                f"Generator checkpoint not found at {generator_path}. Skipping Path B refinement (keeping raw text)."
            )
        else:
            logger.info(f"Loading Generator model from {generator_path}...")
            generator_model = GeneratorModel.from_pretrained(generator_path).to(device)
            generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)
            generator_model.eval()

            # Build Sentence Context Map for fast lookup
            # Group by sentence_id -> list of tokens
            sentence_map = pred_df.groupby("sentence_id")["token"].apply(list).to_dict()

            path_b_indices = pred_df[path_b_mask].index
            input_texts = []

            # Construct context-augmented inputs
            for idx in path_b_indices:
                row = pred_df.loc[idx]
                s_id = row["sentence_id"]

                # Parse token_id from "sentenceId_tokenId"
                try:
                    t_id = int(str(row["id"]).split("_")[-1])
                except ValueError:
                    t_id = 0

                label = row["class"]
                ctx_tokens = sentence_map.get(s_id, [])
                seq_len = len(ctx_tokens)

                if seq_len == 0:
                    # Fallback if context missing
                    input_texts.append(
                        f"{label} {Config.TARGET_START_TOKEN} {row['token']} {Config.TARGET_END_TOKEN}"
                    )
                    continue

                # Define context window
                start = max(0, t_id - Config.CONTEXT_WINDOW_SIZE)
                end = min(seq_len, t_id + Config.CONTEXT_WINDOW_SIZE + 1)

                left_ctx = " ".join(ctx_tokens[start:t_id])
                target_token = ctx_tokens[t_id]
                right_ctx = " ".join(ctx_tokens[t_id + 1 : end])

                input_str = (
                    f"{label} {left_ctx} "
                    f"{Config.TARGET_START_TOKEN} {target_token} {Config.TARGET_END_TOKEN} "
                    f"{right_ctx}"
                )
                input_texts.append(input_str)

            # Batch Generation
            gen_batch_size = Config.GENERATOR_BATCH_SIZE * 4
            generated_outputs = []

            for i in range(0, len(input_texts), gen_batch_size):
                batch_texts = input_texts[i : i + gen_batch_size]

                inputs = generator_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=Config.GENERATOR_MAX_INPUT_LEN,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = generator_model.generate(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        max_length=Config.GENERATOR_MAX_TARGET_LEN,
                    )

                decoded = generator_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                generated_outputs.extend(decoded)

            # Update predictions
            pred_df.loc[path_b_indices, "after"] = generated_outputs

    # ==========================================
    # 5. Save Submission
    # ==========================================
    logger.info(f"Saving submission to {Config.SUBMISSION_FILE}...")
    submission = pred_df[["id", "after"]]
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info("Inference pipeline completed successfully.")
