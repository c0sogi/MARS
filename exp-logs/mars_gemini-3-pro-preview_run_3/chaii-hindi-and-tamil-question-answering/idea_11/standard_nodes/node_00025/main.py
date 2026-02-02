import os
import shutil
import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from library.config import Config
from library.utils import set_seed, compute_jaccard_score, jaccard
from library.tapt_engine import run_tapt_pretraining
from library.qa_engine import QAEngine
from library.inference_engine import InferenceEngine
from library.data import prepare_qa_features, QADataset


def main():
    # 1. Configuration for Fast Baseline
    # Override defaults to ensure completion within strict time limits
    Config.EPOCHS = 3
    Config.TAPT_EPOCHS = 3
    Config.N_FOLDS = 2

    # Initialize directories
    Config.setup()
    set_seed(Config.SEED)

    # Clear cache to ensure fresh feature generation with the TAPT tokenizer
    if os.path.exists(Config.QA_CACHE_DIR):
        shutil.rmtree(Config.QA_CACHE_DIR)
    os.makedirs(Config.QA_CACHE_DIR, exist_ok=True)

    print("=== Starting Pipeline Execution ===")
    Config.print_config()

    # 2. Run TAPT (Task-Adaptive Pretraining)
    print("\n[Step 1] Running TAPT...")
    # We force reload of data to ensure TAPT uses the correct text pairs
    run_tapt_pretraining(load_cached_data=False)

    # 3. Run QA Training (K-Fold)
    print("\n[Step 2] Running QA K-Fold Training...")
    qa_engine = QAEngine()
    qa_engine.run_k_fold_training()

    # 4. Validation on Hold-out Set
    print("\n[Step 3] Validating on Hold-out Set (val.csv)...")

    # Load validation metadata
    val_df = pd.read_csv(Config.VAL_META_PATH)

    # Prepare features for validation
    # We treat it as a test set (is_test=True) to skip label generation for the inference loader,
    # but we will compare predictions against the ground truth in val_df later.
    tokenizer = qa_engine.tokenizer
    val_features = prepare_qa_features(
        tokenizer,
        Config.VAL_META_PATH,
        "val_holdout",
        load_cached_data=False,
        is_test=True,
    )

    val_dataset = QADataset(val_features, is_test=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run Inference with Ensemble
    inference_engine = InferenceEngine()
    all_fold_preds = []

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.QA_MODEL_DIR, f"model_fold_{fold}.pt")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found.")
            continue

        print(f"Predicting with Fold {fold} model on Val set...")
        model = AutoModelForTokenClassification.from_pretrained(
            Config.TAPT_MODEL_DIR, num_labels=3
        )
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        preds = inference_engine.predict_document(model, val_loader, val_features)
        all_fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    # Majority Vote
    if not all_fold_preds:
        print("Error: No predictions generated.")
        return

    final_val_preds = inference_engine.majority_vote(all_fold_preds)

    # Compute Metric
    ground_truths = []
    predictions = []

    # Align predictions with ground truth using IDs
    for _, row in val_df.iterrows():
        eid = row["id"]
        gt = str(row["answer_text"])
        pred = final_val_preds.get(eid, "")

        ground_truths.append(gt)
        predictions.append(pred)

    val_score = compute_jaccard_score(ground_truths, predictions)
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    print("\n[Step 4] Performing Failure Analysis...")

    # Calculate per-sample error (1 - Jaccard)
    sample_scores = [jaccard(gt, pred) for gt, pred in zip(ground_truths, predictions)]
    errors = [1.0 - s for s in sample_scores]

    # Extract features for correlation analysis
    context_lens = val_df["context"].apply(lambda x: len(str(x).split())).tolist()
    question_lens = val_df["question"].apply(lambda x: len(str(x).split())).tolist()

    if len(errors) > 1:
        corr_ctx, _ = pearsonr(errors, context_lens)
        corr_q, _ = pearsonr(errors, question_lens)

        print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
        print(f"Correlation (Error vs Question Length): {corr_q:.4f}")
    else:
        print("Insufficient samples for correlation analysis.")

    # 6. Submission
    THRESHOLD = 0.3414615235510758

    if val_score > THRESHOLD:
        print(
            f"\n[Step 5] Validation score {val_score} > {THRESHOLD}. Generating Submission..."
        )
        inference_engine.ensemble_predict()
    else:
        print(
            f"\n[Step 5] Validation score {val_score} <= {THRESHOLD}. Skipping Submission."
        )


if __name__ == "__main__":
    main()
