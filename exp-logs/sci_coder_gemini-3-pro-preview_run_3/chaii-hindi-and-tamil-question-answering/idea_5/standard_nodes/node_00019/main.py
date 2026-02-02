import sys
import os
import pandas as pd
import numpy as np
import torch
from collections import Counter

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, jaccard
from library.tapt_engine import run_tapt
from library.qa_engine import run_training
from library.inference_engine import InferenceEngine
from library.data_factory import get_dataloader


def main():
    # 1. Setup
    seed_everything(42)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Task-Adaptive Pretraining (TAPT)
    # Adapts the base model to the domain text.
    print("Step 1: Running Task-Adaptive Pretraining (TAPT)...")
    try:
        run_tapt(load_cached_data=True)
    except Exception as e:
        print(f"Warning: TAPT execution encountered an issue: {e}")
        print(
            "Proceeding, assuming base model or previous checkpoints might be available."
        )

    # 3. QA Training
    # Fine-tunes the model on the QA dataset for multiple seeds.
    print("\nStep 2: Running QA Training...")
    run_training(load_cached_data=True)

    # 4. Validation and Scoring
    print("\nStep 3: Validation and Scoring...")

    # Load Validation Metadata (Ground Truth)
    if not os.path.exists(Config.VAL_META_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_META_PATH}"
        )

    val_df = pd.read_csv(Config.VAL_META_PATH)
    val_df["id"] = val_df["id"].astype(str)
    val_df["answer_text"] = val_df["answer_text"].fillna("").astype(str)

    # Map ID to Ground Truth Answer
    gt_map = dict(zip(val_df["id"], val_df["answer_text"]))

    # Get Validation DataLoader
    val_loader = get_dataloader(
        mode="val",
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        load_cached_data=True,
    )

    # Initialize Inference Engine
    engine = InferenceEngine()

    # Collect predictions from all seeded models
    all_model_preds = []
    for seed in Config.SEEDS:
        print(f"Generating predictions for Seed {seed}...")
        try:
            model = engine.load_model(seed)
            preds = engine.predict_single_model(model, val_loader)
            all_model_preds.append(preds)

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
        except FileNotFoundError:
            print(f"Warning: Model for seed {seed} not found. Skipping.")

    if not all_model_preds:
        print("Error: No models available for validation.")
        print("Final Validation Metric: 0.0")
        return

    # Ensemble: Majority Vote
    print("Aggregating predictions (Majority Vote)...")
    final_preds_map = {}

    for ex_id in val_df["id"]:
        votes = []
        for model_preds in all_model_preds:
            # Default to empty string if model didn't predict for this ID
            votes.append(model_preds.get(ex_id, ""))

        # Select the most common prediction string
        counts = Counter(votes)
        best_answer, _ = counts.most_common(1)[0]
        final_preds_map[ex_id] = best_answer

    # Compute Jaccard Scores
    jaccard_scores = []
    for ex_id in val_df["id"]:
        prediction = final_preds_map.get(ex_id, "")
        ground_truth = gt_map.get(ex_id, "")
        score = jaccard(ground_truth, prediction)
        jaccard_scores.append(score)

    avg_jaccard = np.mean(jaccard_scores)

    # Print Metric (Required Format)
    print(f"Final Validation Metric: {avg_jaccard}")

    # 5. Failure Analysis
    print("\nStep 4: Failure Analysis...")

    # Add analysis columns
    val_df["prediction"] = val_df["id"].map(final_preds_map)
    val_df["jaccard"] = jaccard_scores
    val_df["error"] = 1.0 - val_df["jaccard"]

    # Compute Input Features
    # Context Length (words)
    val_df["context_len"] = (
        val_df["context"].fillna("").astype(str).apply(lambda x: len(x.split()))
    )
    # Question Length (words)
    val_df["question_len"] = (
        val_df["question"].fillna("").astype(str).apply(lambda x: len(x.split()))
    )

    # Calculate Correlations
    features_to_analyze = ["context_len", "question_len"]

    print("Correlations with Error Magnitude (1 - Jaccard):")
    error_std = val_df["error"].std()

    for feat in features_to_analyze:
        feat_std = val_df[feat].std()

        # Ensure variance exists to compute correlation
        if feat_std > 1e-9 and error_std > 1e-9:
            corr = np.corrcoef(val_df["error"], val_df[feat])[0, 1]
            print(f"Correlation between Error and {feat}: {corr:.4f}")
        else:
            print(f"Correlation between Error and {feat}: Undefined (Zero Variance)")

    # 6. Submission Generation
    print("\nStep 5: Submission Generation...")
    THRESHOLD = 0.3011529653320698

    if avg_jaccard > THRESHOLD:
        print(f"Validation metric ({avg_jaccard}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        engine.generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric ({avg_jaccard}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
