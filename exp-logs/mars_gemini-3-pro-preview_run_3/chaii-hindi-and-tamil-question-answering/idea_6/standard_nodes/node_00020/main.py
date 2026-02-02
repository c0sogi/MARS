import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

from library.configuration import Config
from library.utilities import set_seed, compute_average_jaccard, jaccard
from library.tapt_engine import run_tapt_pretraining
from library.qa_training_engine import run_qa_training
from library.qa_inference_engine import QAInferenceEngine, generate_submission
from library.qa_data_processing import prepare_eval_features, QADataset


def run_validation_and_analysis():
    """
    Performs inference on the validation set, computes the Jaccard metric,
    and runs failure analysis.
    """
    print("\n--- Starting Validation & Failure Analysis ---")

    # 1. Load Validation Data
    # We need to load the raw metadata to get ground truths and text for analysis
    df_val = pd.read_csv(Config.VAL_FILE)

    # Limit for debugging if configured
    if Config.MAX_VAL_SAMPLES:
        df_val = df_val.iloc[: Config.MAX_VAL_SAMPLES]

    # 2. Prepare Validation Data for Inference
    # We need to process it in 'eval' mode (sliding windows, retaining example_ids)
    # rather than 'train' mode (labels).

    # Determine which tokenizer to use (TAPT or Base)
    if os.path.exists(Config.TAPT_OUTPUT_DIR):
        tokenizer = AutoTokenizer.from_pretrained(Config.TAPT_OUTPUT_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    print("Generating validation features for inference...")
    val_features = prepare_eval_features(df_val, tokenizer)
    val_ds = QADataset(val_features, is_train=False)

    # 3. Run Inference (Ensemble)
    all_predictions = []
    for seed in Config.SEEDS:
        print(f"Predicting validation set with Seed {seed}...")
        preds = QAInferenceEngine.predict_single_model(seed, val_ds, tokenizer)
        if preds:
            all_predictions.append(preds)

    if not all_predictions:
        print("Error: No predictions generated for validation.")
        return 0.0

    # Majority Vote
    final_preds_map = QAInferenceEngine.majority_vote(all_predictions)

    # 4. Compute Metric
    ground_truths = []
    predictions = []
    ids = []

    # Align predictions with ground truth dataframe
    for _, row in df_val.iterrows():
        eid = row["id"]
        gt = row["answer_text"]
        pred = final_preds_map.get(eid, "")

        ids.append(eid)
        ground_truths.append(gt)
        predictions.append(pred)

    final_metric = compute_average_jaccard(ground_truths, predictions)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate per-sample Jaccard and Error
    sample_scores = [jaccard(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    errors = [1.0 - s for s in sample_scores]

    # Create Analysis DataFrame
    analysis_df = df_val.copy()
    analysis_df["jaccard"] = sample_scores
    analysis_df["error"] = errors

    # Feature Engineering for Analysis
    analysis_df["context_len"] = analysis_df["context"].apply(len)
    analysis_df["question_len"] = analysis_df["question"].apply(len)
    analysis_df["is_tamil"] = (analysis_df["language"] == "tamil").astype(int)

    # Compute Correlations
    correlations = analysis_df[
        ["error", "context_len", "question_len", "is_tamil"]
    ].corr()["error"]

    print("Correlation between Error and Input Features:")
    print(correlations.drop("error"))  # Print correlations with other features

    return final_metric


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.create_directories()

    print("=== Starting Pipeline ===")

    # 2. Task-Adaptive Pretraining (TAPT)
    # Adapts the model to the domain vocabulary
    print("\n=== Step 1: Task-Adaptive Pretraining ===")
    run_tapt_pretraining(load_cached_data=True)

    # 3. QA Training
    # Fine-tunes the model using stratified sampling and ensembling
    print("\n=== Step 2: QA Fine-Tuning ===")
    run_qa_training(load_cached_data=True)

    # 4. Validation and Analysis
    # Evaluates performance and checks for systematic errors
    print("\n=== Step 3: Validation ===")
    val_score = run_validation_and_analysis()

    # 5. Submission
    # Generates submission file if validation score meets threshold
    THRESHOLD = 0.3011529653320698
    print(f"\nValidation Score: {val_score} (Threshold: {THRESHOLD})")

    if val_score > THRESHOLD:
        print("\n=== Step 4: Generating Submission ===")
        generate_submission()
    else:
        print(
            "\nValidation score did not meet threshold. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
