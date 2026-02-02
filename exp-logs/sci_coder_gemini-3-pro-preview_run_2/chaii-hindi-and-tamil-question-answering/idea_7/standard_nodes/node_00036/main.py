import pandas as pd
import numpy as np
import torch
import gc
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import get_data, QADataset
from library.model import XLMRobertaForMultiTaskQA
from library.engine import train_one_epoch, validate
from library.postprocessing import postprocess_predictions, save_submission


def main():
    # 1. Initialization
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load training data
    print("Loading training data...")
    train_features = get_data(split="train")
    train_dataset = QADataset(train_features, is_test=False)

    # Load validation data
    print("Loading validation data...")
    val_features = get_data(split="val")
    val_dataset = QADataset(val_features, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Test Data
    print("Loading test data...")
    test_features = get_data(split="test")
    test_dataset = QADataset(test_features, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulators for logits
    val_logits_accum = None
    test_logits_accum = None

    # Loop over seeds
    for i, seed in enumerate(Config.SEEDS):
        print(f"\n=== Starting Run with Seed {seed} ({i+1}/{len(Config.SEEDS)}) ===")
        seed_everything(seed)

        # Create Train Loader (re-shuffled by seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Model Setup
        print(f"Initializing model: {Config.MODEL_NAME}")
        model = XLMRobertaForMultiTaskQA()
        model.to(device)

        # Optimizer and Scheduler
        # Differential Learning Rates: Higher LR for heads, lower for backbone
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            # Backbone parameters (lower LR)
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "roberta" in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.LEARNING_RATE,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "roberta" in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.LEARNING_RATE,
            },
            # Head parameters (higher LR - 3x)
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "roberta" not in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.LEARNING_RATE * 3,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "roberta" not in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.LEARNING_RATE * 3,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        # Adjust total steps for gradient accumulation
        num_update_steps_per_epoch = (
            len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
        )
        total_steps = num_update_steps_per_epoch * Config.EPOCHS
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # 4. Training Loop
        print("Starting training...")
        for epoch in range(Config.EPOCHS):
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
            train_loss = train_one_epoch(
                model, optimizer, scheduler, train_loader, device, epoch
            )
            print(f"Average Training Loss: {train_loss:.4f}")

        # 5. Inference
        print("Running validation inference...")
        val_preds = validate(model, val_loader, device)

        print("Running test inference...")
        test_preds = validate(model, test_loader, device)

        # Accumulate
        if val_logits_accum is None:
            val_logits_accum = {k: v for k, v in val_preds.items() if k != "loss"}
            test_logits_accum = {k: v for k, v in test_preds.items() if k != "loss"}
        else:
            for k in val_logits_accum:
                val_logits_accum[k] += val_preds[k]
                test_logits_accum[k] += test_preds[k]

        # Cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # Average Logits
    n_seeds = len(Config.SEEDS)
    val_predictions_raw = {k: v / n_seeds for k, v in val_logits_accum.items()}
    test_predictions_raw = {k: v / n_seeds for k, v in test_logits_accum.items()}

    # 6. Evaluation
    # Load original validation metadata to get ground truth text and context
    val_meta = pd.read_csv(Config.VAL_META)

    # Convert logits to text predictions
    print("Post-processing validation predictions...")
    val_submission_df = postprocess_predictions(
        val_meta, val_features, val_predictions_raw
    )

    # Merge predictions with ground truth for scoring
    val_merged = val_meta.merge(val_submission_df, on="id", how="left")
    val_merged["PredictionString"] = val_merged["PredictionString"].fillna("")

    # Calculate Jaccard Score
    scores = []
    for _, row in val_merged.iterrows():
        gt = str(row["answer_text"])
        pred = str(row["PredictionString"])
        scores.append(jaccard(gt, pred))

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_merged["jaccard_score"] = scores
    val_merged["error"] = 1.0 - val_merged["jaccard_score"]

    # Feature extraction for correlation
    val_merged["context_len"] = val_merged["context"].astype(str).apply(len)
    val_merged["question_len"] = val_merged["question"].astype(str).apply(len)

    # Compute correlations
    corr_context = val_merged["error"].corr(val_merged["context_len"])
    corr_question = val_merged["error"].corr(val_merged["question_len"])

    print(f"Correlation Error vs Context Length: {corr_context}")
    print(f"Correlation Error vs Question Length: {corr_question}")

    # 8. Submission
    THRESHOLD = 0.5683333333333334

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Post-processing
        test_meta = pd.read_csv(Config.TEST_META)
        submission_df = postprocess_predictions(
            test_meta, test_features, test_predictions_raw
        )

        # Save
        save_submission(submission_df)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
