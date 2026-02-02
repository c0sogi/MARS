import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import provided library components
from library.config import Config
from library.utils import set_seed, jaccard
from library.data_processing import get_qa_data, qa_collate_fn
from library.tapt_engine import run_tapt_training
from library.qa_engine import run_qa_training
from library.inference_engine import (
    predict_single_model,
    ensemble_predictions,
    generate_submission,
)


def main():
    # 1. Setup and Initialization
    print("Initializing Experiment...")
    Config.setup()
    set_seed(42)

    # 2. Task-Adaptive Pretraining (TAPT)
    # Adapts the base model to the domain text
    tapt_model_dir = run_tapt_training()

    # 3. QA Training
    # Trains an ensemble of models using the TAPT weights
    run_qa_training(tapt_model_dir=tapt_model_dir)

    # 4. Validation & Failure Analysis
    print("\n" + "=" * 30)
    print("Running Validation & Failure Analysis")
    print("=" * 30)

    # Load Validation Data
    # We use the tokenizer from the TAPT model (which is the base for QA)
    tokenizer = AutoTokenizer.from_pretrained(tapt_model_dir)
    _, val_ds, _ = get_qa_data(tokenizer, load_cached_data=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device(Config.DEVICE)
    all_model_preds = []

    # Generate predictions for each seed model on the validation set
    for seed in Config.SEED_LIST:
        model_path = os.path.join(Config.QA_MODELS_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        print(f"Evaluating model seed {seed} on validation set...")

        # Load Model
        model = AutoModelForTokenClassification.from_pretrained(
            tapt_model_dir, num_labels=3
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Predict
        preds = predict_single_model(model, val_loader, tokenizer, device)
        all_model_preds.append(preds)

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # Ensemble Predictions
    print("Ensembling validation predictions...")
    final_val_preds = ensemble_predictions(all_model_preds)

    # Calculate Metric and Prepare Analysis Data
    val_meta_df = pd.read_csv(Config.VAL_META_PATH)

    jaccard_scores = []
    analysis_records = []

    for _, row in val_meta_df.iterrows():
        eid = row["id"]
        ground_truth = str(row["answer_text"])
        prediction = final_val_preds.get(eid, "")

        score = jaccard(ground_truth, prediction)
        jaccard_scores.append(score)

        # Collect features for failure analysis
        analysis_records.append(
            {
                "error": 1.0 - score,
                "context_length": len(str(row["context"])),
                "question_length": len(str(row["question"])),
                "answer_length": len(ground_truth),
            }
        )

    final_metric = np.mean(jaccard_scores)

    # Required Output Format
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\n--- Failure Analysis ---")
    analysis_df = pd.DataFrame(analysis_records)
    if not analysis_df.empty:
        correlations = analysis_df.corr()["error"].drop("error")
        print("Correlation with Error Magnitude:")
        print(correlations)
    else:
        print("No validation data available for analysis.")

    # 5. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.3011529653320698

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
