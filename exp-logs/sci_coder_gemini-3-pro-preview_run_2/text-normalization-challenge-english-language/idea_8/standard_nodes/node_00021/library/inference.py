import os
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm

from library.config import cfg
from library.data_utils import (
    load_test_router_data,
    prepare_generator_inference_data,
    get_router_tokenizer,
)
from library.modeling import RouterModel, GeneratorModel
from library.normalization_rules import dispatch_rule


def align_router_predictions(dataset, df_test, raw_preds):
    """
    Aligns subword predictions from the Router model to token-level labels.
    Ensures that the flat list of predictions corresponds to the sentence-grouped structure.
    """
    tokenizer = get_router_tokenizer()
    encodings = dataset.encodings

    # We iterate over unique sentence IDs in sorted order, as this matches
    # the grouping logic in data_utils.process_router_data
    unique_sentence_ids = sorted(df_test["sentence_id"].unique())

    # Pre-group dataframe for fallback access if needed
    grouped_df = df_test.groupby("sentence_id")

    aligned_preds = []

    print("Aligning Router predictions to tokens...")
    for i, sent_id in enumerate(tqdm(unique_sentence_ids, desc="Aligning")):
        if i >= len(raw_preds):
            break

        # Get subword predictions for this sentence (List[int])
        seq_preds = raw_preds[i]

        # Get word_ids mapping to align subwords to tokens
        # We try to access the batch encoding directly
        try:
            word_ids = encodings.word_ids(batch_index=i)
        except (AttributeError, KeyError):
            # Fallback: Re-tokenize if the dataset object doesn't expose word_ids directly
            # This ensures robustness against internal dataset structure changes
            group = grouped_df.get_group(sent_id)
            text = group["before"].astype(str).tolist()
            # Must match the tokenizer config used in training/processing
            tmp = tokenizer(
                text,
                is_split_into_words=True,
                truncation=True,
                max_length=cfg.MAX_LENGTH_ROUTER,
            )
            word_ids = tmp.word_ids()

        # Map to tokens: Strategy is to take the prediction of the first subword
        token_preds = []
        seen_tokens = set()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue

            # If this is the first time we see this token index, record the prediction
            if word_idx not in seen_tokens:
                if idx < len(seq_preds):
                    token_preds.append(seq_preds[idx])
                else:
                    # If prediction was truncated, default to PLAIN
                    token_preds.append(cfg.CLASS2ID["PLAIN"])
                seen_tokens.add(word_idx)

        aligned_preds.append(token_preds)

    return aligned_preds


def run_inference():
    """
    Main execution pipeline for generating the submission.
    """
    # ==========================================
    # 1. Load Test Data & Router
    # ==========================================
    print("Loading test data...")
    # df_test contains the raw metadata; router_dataset contains tokenized inputs
    router_dataset, df_test = load_test_router_data()

    print("Initializing Router Model...")
    router = RouterModel()
    router.load_model("router_best")

    # ==========================================
    # 2. Router Inference (Class Prediction)
    # ==========================================
    print("Running Router inference...")
    # raw_router_preds is a list of lists (sentence_idx -> sequence of class IDs)
    raw_router_preds = router.predict(router_dataset)

    # Align subword predictions to token level
    aligned_preds = align_router_predictions(router_dataset, df_test, raw_router_preds)

    # Ensure predictions match the exact token count of the dataframe
    # (Fixes potential truncation or padding mismatches)
    print("Finalizing class alignment...")
    sent_sizes = df_test.groupby("sentence_id").size()
    sorted_sent_ids = sorted(df_test["sentence_id"].unique())

    final_preds = []
    for sent_id, preds in zip(sorted_sent_ids, aligned_preds):
        target_len = sent_sizes[sent_id]

        if len(preds) < target_len:
            # Pad missing predictions with PLAIN
            preds.extend([cfg.CLASS2ID["PLAIN"]] * (target_len - len(preds)))
        elif len(preds) > target_len:
            # Clip excess predictions
            preds = preds[:target_len]

        final_preds.append(preds)

    # Free up Router memory
    del router, router_dataset, raw_router_preds
    torch.cuda.empty_cache()

    # ==========================================
    # 3. Generator Inference (Complex Classes)
    # ==========================================
    print("Preparing Generator inputs...")
    # Prepare data only for tokens classified as Neural classes (DATE, TIME, etc.)
    # Returns dataset and metadata mapping (sent_id, token_id) -> generated_text
    gen_dataset, gen_metadata = prepare_generator_inference_data(df_test, final_preds)

    gen_results = {}
    if gen_dataset is not None and len(gen_dataset) > 0:
        print(f"Running Generator inference on {len(gen_dataset)} complex tokens...")
        generator = GeneratorModel()
        generator.load_model("generator_best")

        gen_texts = generator.predict(gen_dataset)

        # Store results in a lookup dictionary
        for (sid, tid), text in zip(gen_metadata, gen_texts):
            gen_results[(sid, tid)] = text

        del generator, gen_dataset
        torch.cuda.empty_cache()
    else:
        print("No tokens routed to Generator.")

    # ==========================================
    # 4. Hybrid Assembly & Submission
    # ==========================================
    print("Applying hybrid normalization logic...")

    # Create a mapping for fast access: sentence_id -> list of class_ids
    pred_map = {sid: preds for sid, preds in zip(sorted_sent_ids, final_preds)}

    submission_rows = []

    # Convert dataframe to dict records for faster iteration than itertuples
    records = df_test.to_dict("records")

    for row in tqdm(records, desc="Generating Submission"):
        sid = row["sentence_id"]
        tid = row["token_id"]
        text = str(row["before"])
        row_id = row["id"]

        # Retrieve predicted class
        # Fallback to PLAIN if indices are somehow misaligned (safety check)
        try:
            p_id = pred_map[sid][tid]
            cls_name = cfg.ID2CLASS[p_id]
        except (KeyError, IndexError):
            cls_name = "PLAIN"

        normalized = text

        # Route to appropriate handler
        if cls_name in cfg.NEURAL_BASED_CLASSES:
            # Path B: Neural Generator
            if (sid, tid) in gen_results:
                normalized = gen_results[(sid, tid)]
            else:
                # Fallback: If generator didn't run for this token (e.g. context edge case),
                # keep raw text or try rule dispatch. Identity is safest fallback.
                normalized = text
        else:
            # Path A: Deterministic Rules
            normalized = dispatch_rule(text, cls_name)

        submission_rows.append({"id": row_id, "after": normalized})

    # Save to CSV
    submission_df = pd.DataFrame(submission_rows)
    out_path = os.path.join(cfg.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved successfully to {out_path}")
