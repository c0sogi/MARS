import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, jaccard
from library.data import QADataset, qa_collate_fn
from library.model import WeightedTokenClassifier
from library.tapt_engine import run_tapt
from library.qa_engine import train_model, get_predictions
from library.inference import run_inference, ensemble_vote


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Starting pipeline on device: {device}")

    # 2. Task-Adaptive Pretraining (TAPT)
    # Adapts the backbone model to the specific linguistic domain (Hindi/Tamil)
    print("\n=== Step 1: Task-Adaptive Pretraining ===")
    run_tapt()

    # 3. Training Loop (Ensemble)
    # Train independent models for each seed defined in Config
    print("\n=== Step 2: Training Ensemble Models ===")
    for seed in Config.SEEDS:
        train_model(seed)

    # 4. Validation and Failure Analysis
    print("\n=== Step 3: Validation & Failure Analysis ===")

    # Load Validation Dataset
    val_dataset = QADataset(mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Ground Truth for Evaluation
    val_df = pd.read_csv(Config.VAL_CSV)
    # Ensure answer text is string and handle NaNs
    val_df["answer_text"] = val_df["answer_text"].fillna("").astype(str)
    val_ids = val_df["id"].tolist()

    # Generate predictions from all trained seeds
    all_seed_predictions = []

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        print(f"Generating validation predictions for seed {seed}...")

        # Load Model
        model = WeightedTokenClassifier(class_weights=None)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Inference
        with torch.no_grad():
            preds = get_predictions(model, val_loader, device)
        all_seed_predictions.append(preds)

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    # Aggregate predictions using Majority Voting
    final_val_preds = ensemble_vote(all_seed_predictions, val_ids)

    # Calculate Metrics and Error Analysis Features
    scores = []
    errors = []
    context_lens = []
    question_lens = []

    for _, row in val_df.iterrows():
        eid = row["id"]
        gt_text = row["answer_text"]
        pred_text = final_val_preds.get(eid, "")

        # Calculate Jaccard Score
        score = jaccard(gt_text, pred_text)
        scores.append(score)

        # Data for failure analysis
        errors.append(1.0 - score)
        # Simple whitespace tokenization for length analysis
        context_lens.append(len(str(row["context"]).split()))
        question_lens.append(len(str(row["question"]).split()))

    # Compute and print final metric
    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlations for Failure Analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "context_len": context_lens, "question_len": question_lens}
    )

    corr_ctx = df_analysis["error"].corr(df_analysis["context_len"])
    corr_q = df_analysis["error"].corr(df_analysis["question_len"])

    print("\nFailure Analysis Correlations:")
    print(f"Error vs Context Length: {corr_ctx}")
    print(f"Error vs Question Length: {corr_q}")

    # 5. Submission Generation
    # Only submit if the model meets the performance threshold
    THRESHOLD = 0.3011529653320698

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
