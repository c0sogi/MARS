import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, jaccard, postprocess_qa_predictions
from library.data import load_data, QADataset, get_tokenizer
from library.model import QAModel
from library.engine import run_training, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_failure_analysis():
    """
    Loads the best model, performs inference on the validation set,
    computes the Jaccard metric, and analyzes correlations between
    error magnitude and input features.
    """
    print("\n=== Starting Failure Analysis ===")

    device = Config.DEVICE
    tokenizer = get_tokenizer()

    # Load Validation Data
    # Use cached features if available to speed up loading
    val_features = load_data("val", tokenizer, load_cached_data=True, debug=False)
    val_examples = pd.read_csv(Config.VAL_DATA_PATH)

    # Load Ensemble Models
    models = []
    for seed in Config.SEEDS:
        model_path = os.path.join(Config.OUTPUT_DIR, f"best_model_seed_{seed}.pth")
        if os.path.exists(model_path):
            m = QAModel(Config.MODEL_CHECKPOINT)
            m.load_state_dict(torch.load(model_path, map_location=device))
            m.to(device)
            m.eval()
            models.append(m)

    if not models:
        print("Error: No models found for ensemble analysis.")
        return

    # Prepare DataLoader for Inference
    # mode="val" ensures we get input_ids and attention_mask
    val_dataset = QADataset(val_features, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference (Ensemble)
    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            batch_start_logits = []
            batch_end_logits = []

            for model in models:
                s, e = model(input_ids, attention_mask)
                batch_start_logits.append(s.cpu().numpy())
                batch_end_logits.append(e.cpu().numpy())

            # Average logits
            avg_start = np.mean(batch_start_logits, axis=0)
            avg_end = np.mean(batch_end_logits, axis=0)

            all_start_logits.append(avg_start)
            all_end_logits.append(avg_end)

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process logits to get text predictions
    # Convert Pandas DataFrames to HuggingFace Datasets for compatibility with utils
    hf_examples = Dataset.from_pandas(val_examples)
    hf_features = Dataset.from_pandas(val_features)

    predictions = postprocess_qa_predictions(
        examples=hf_examples,
        features=hf_features,
        predictions=(all_start_logits, all_end_logits),
        n_best_size=Config.N_BEST_SIZE,
        max_answer_length=Config.MAX_ANSWER_LENGTH,
    )

    # Compute Metrics and Correlations
    scores = []
    errors = []
    context_lens = []
    question_lens = []

    # Create lookup maps for ground truth and features
    gt_map = dict(zip(val_examples["id"], val_examples["answer_text"]))
    ctx_map = dict(zip(val_examples["id"], val_examples["context"]))
    q_map = dict(zip(val_examples["id"], val_examples["question"]))

    for ex_id, pred_text in predictions.items():
        if ex_id in gt_map:
            gt_text = gt_map[ex_id]

            # Compute Jaccard Score
            score = jaccard(gt_text, pred_text)
            scores.append(score)

            # Error Magnitude (0 to 1)
            errors.append(1.0 - score)

            # Extract Features
            ctx_text = str(ctx_map.get(ex_id, ""))
            q_text = str(q_map.get(ex_id, ""))

            context_lens.append(len(ctx_text))
            question_lens.append(len(q_text))

    # Print Final Metric (Required Format)
    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # Correlation Analysis
    if len(scores) > 1:
        df_analysis = pd.DataFrame(
            {
                "error": errors,
                "context_len": context_lens,
                "question_len": question_lens,
            }
        )

        # Calculate Pearson correlation
        corr_ctx = df_analysis["error"].corr(df_analysis["context_len"])
        corr_q = df_analysis["error"].corr(df_analysis["question_len"])

        print("\nCorrelation between Error Magnitude and Input Features:")
        print(f"Context Length vs Error: {corr_ctx:.10f}")
        print(f"Question Length vs Error: {corr_q:.10f}")
    else:
        print("Not enough samples for correlation analysis.")

    return final_metric


def main():
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)

    # 2. Run Training
    # debug=False uses the full provided metadata splits (approx 800 train samples)
    print("=== Starting Training Pipeline ===")
    run_training(debug=False)

    # 3. Evaluation & Failure Analysis
    # This step loads the best model and computes the final metric and correlations
    final_metric = perform_failure_analysis()

    # 4. Generate Submission
    # Predicts on test set and saves to submission.csv
    PREVIOUS_BEST = 0.2522202380952381

    if final_metric > PREVIOUS_BEST:
        print(
            f"\nMetric ({final_metric:.5f}) > Previous Best ({PREVIOUS_BEST:.5f}). Generating Submission..."
        )
        print("\n=== Generating Submission ===")
        generate_submission(debug=False)
    else:
        print(
            f"\nMetric ({final_metric:.5f}) <= Previous Best ({PREVIOUS_BEST:.5f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
