import pandas as pd
import numpy as np
import torch
import os
import sys

# Import provided library modules
from library.config import cfg
from library import (
    train_router,
    train_generator,
    inference,
    modeling,
    data_utils,
    normalization_rules,
)


def main():
    # 1. Setup and Configuration
    cfg.seed_everything()

    # Optimize for fast baseline execution within time limits
    # We reduce epochs to 1. DeBERTa and ByT5 converge relatively quickly.
    cfg.ROUTER_EPOCHS = 1
    cfg.GENERATOR_EPOCHS = 1

    # 2. Model Training
    # Train Router (Contextual Classifier)
    # debug=False ensures we use the full dataset (with downsampling) for good performance
    train_router.run_router_training(debug=False, load_cached_data=True)

    # Train Generator (Seq2Seq for complex classes)
    train_generator.run_generator_training(debug=False, load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("Running Validation and Failure Analysis...")
    val_metric = perform_validation()

    print(f"Final Validation Metric: {val_metric}")

    # 4. Submission Generation
    # Threshold defined in task description
    SUBMISSION_THRESHOLD = 0.9906485140019942

    if val_metric > SUBMISSION_THRESHOLD:
        print("Metric threshold met. Generating submission...")
        inference.run_inference()
    else:
        print(
            f"Metric {val_metric} did not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


def perform_validation():
    """
    Runs inference on the validation set, computes accuracy, and performs failure analysis.
    """
    # Load Validation Data
    df_val = pd.read_csv(cfg.VAL_FILE, keep_default_na=False)

    # Load processed router data for validation
    # This uses the cache generated during training (or creates it if missing)
    val_router_ds = data_utils.load_router_data(split="val", load_cached_data=True)

    # --- Router Inference ---
    router = modeling.RouterModel()
    router.load_model("router_best")

    # Get raw class ID predictions
    raw_preds = router.predict(val_router_ds)

    # Align predictions to token level
    aligned_preds = inference.align_router_predictions(val_router_ds, df_val, raw_preds)

    # Clean up Router
    del router
    torch.cuda.empty_cache()

    # --- Prepare for Generator/Hybrid Logic ---
    sorted_sent_ids = sorted(df_val["sentence_id"].unique())
    sent_sizes = df_val.groupby("sentence_id").size()

    # Process predictions to match dataframe structure (padding/clipping)
    final_preds_list = []
    for i, sent_id in enumerate(sorted_sent_ids):
        preds = aligned_preds[i]
        target_len = sent_sizes[sent_id]

        if len(preds) < target_len:
            preds.extend([cfg.CLASS2ID["PLAIN"]] * (target_len - len(preds)))
        elif len(preds) > target_len:
            preds = preds[:target_len]
        final_preds_list.append(preds)

    # Map sentence_id to predictions for fast lookup
    pred_map = {sid: preds for sid, preds in zip(sorted_sent_ids, final_preds_list)}

    # --- Generator Inference ---
    # Prepare inputs for tokens classified as Neural classes
    gen_dataset, gen_metadata = data_utils.prepare_generator_inference_data(
        df_val, final_preds_list
    )

    gen_results = {}
    if gen_dataset is not None and len(gen_dataset) > 0:
        generator = modeling.GeneratorModel()
        generator.load_model("generator_best")
        gen_texts = generator.predict(gen_dataset)

        for (sid, tid), text in zip(gen_metadata, gen_texts):
            gen_results[(sid, tid)] = text

        del generator
        torch.cuda.empty_cache()

    # --- Scoring & Failure Analysis ---
    correct_count = 0
    total_count = len(df_val)

    errors = []
    token_lengths = []

    # Iterate through validation data to normalize and compare
    # Using itertuples for performance
    for row in df_val.itertuples(index=False):
        sid = row.sentence_id
        tid = row.token_id
        text = str(row.before)
        true_after = str(row.after)

        # Get predicted class
        try:
            p_id = pred_map[sid][tid]
            cls_name = cfg.ID2CLASS[p_id]
        except (KeyError, IndexError):
            cls_name = "PLAIN"

        # Normalize based on class
        if cls_name in cfg.NEURAL_BASED_CLASSES:
            # Path B: Neural Generator
            if (sid, tid) in gen_results:
                normalized = gen_results[(sid, tid)]
            else:
                # Fallback
                normalized = text
        else:
            # Path A: Rules
            normalized = normalization_rules.dispatch_rule(text, cls_name)

        # Compare
        if normalized == true_after:
            correct_count += 1
            errors.append(0)
        else:
            errors.append(1)

        token_lengths.append(len(text))

    # Compute Metric
    accuracy = correct_count / total_count

    # Failure Analysis: Correlation
    if len(errors) > 0:
        corr = np.corrcoef(errors, token_lengths)[0, 1]
        print(f"Correlation (Error vs Token Length): {corr}")
    else:
        print("Correlation (Error vs Token Length): N/A (No data)")

    return accuracy


if __name__ == "__main__":
    main()
