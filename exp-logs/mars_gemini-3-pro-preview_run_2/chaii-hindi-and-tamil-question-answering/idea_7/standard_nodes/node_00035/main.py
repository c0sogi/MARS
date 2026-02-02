import pandas as pd
import numpy as np
import torch
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
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load training data
    print("Loading training data...")
    train_features = get_data(split="train")
    train_dataset = QADataset(train_features, is_test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

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

    # 3. Model Setup
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = XLMRobertaForMultiTaskQA()
    model.to(device)

    # Optimizer and Scheduler
    # Differential Learning Rates (Cite solution_lesson_node_00034)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "roberta" in n],
            "lr": Config.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model.named_parameters() if "roberta" not in n],
            "lr": Config.LR_HEAD,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY)

    # Adjust total steps for gradient accumulation
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
    total_steps = num_update_steps_per_epoch * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )
        print(f"Average Training Loss: {train_loss:.4f}")

    # 5. Validation & Evaluation
    print("\nRunning validation inference...")
    # Get raw logits from the model
    val_predictions_raw = validate(model, val_loader, device)

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

    # 6. Failure Analysis
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

    # 7. Submission
    THRESHOLD = 0.5683333333333334

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_features = get_data(split="test")
        test_dataset = QADataset(test_features, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_predictions_raw = validate(model, test_loader, device)

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
