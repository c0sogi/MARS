import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, jaccard, post_process_predictions
from library.data import get_processed_data, QADataset
from library.model import CustomXLMRoberta
from library.engine import get_optimizer_grouped_parameters, train_fn, inference_fn


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("=== Setting up Demo Configuration ===")
    set_seed(42)

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small subset for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 2
    Config.VALID_BATCH_SIZE = 2
    Config.GRAD_ACCUM_STEPS = 1
    Config.MODEL_CHECKPOINT = (
        "xlm-roberta-base"  # Use base model for faster loading in demo
    )
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_CHECKPOINT}")

    # 2. Data Loading and Processing
    print("\n=== Testing Data Pipeline ===")
    # Load Training Data
    print("Loading Train Data...")
    train_ds, train_features = get_processed_data("train", debug=True)

    # Validation: Ensure train_ds is a QADataset and has length
    assert isinstance(
        train_ds, QADataset
    ), "train_ds should be an instance of QADataset"
    assert len(train_ds) > 0, "Training dataset is empty"
    print(f"Train Dataset Size: {len(train_ds)}")

    # Check a single item
    sample_item = train_ds[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "answerable",
    ]
    assert all(
        k in sample_item for k in required_keys
    ), f"Missing keys in dataset item: {sample_item.keys()}"
    print("Train item keys verified.")

    # Load Validation Data
    print("Loading Validation Data...")
    val_ds, val_features = get_processed_data("val", debug=True)
    assert len(val_ds) > 0, "Validation dataset is empty"
    print(f"Validation Dataset Size: {len(val_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in simple demo
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = CustomXLMRoberta(config_path=Config.MODEL_CHECKPOINT, pretrained=True)
    model.to(Config.DEVICE)

    # Verify model structure
    assert hasattr(model, "qa_outputs"), "Model missing QA head"
    assert hasattr(
        model, "answerability_classifier"
    ), "Model missing Answerability head"
    print("Model initialized successfully.")

    # 4. Optimizer and Scheduler Setup
    print("\n=== Setting up Optimizer ===")
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )
    print("Optimizer and Scheduler ready.")

    # 5. Training Loop Demonstration
    print("\n=== Running Training Step ===")
    # Run just one epoch (which is very short due to DEBUG_SAMPLE_SIZE)
    avg_loss = train_fn(
        train_loader, model, optimizer, Config.DEVICE, scheduler, epoch=0
    )

    assert not np.isnan(avg_loss), "Training loss returned NaN"
    print(f"Training step complete. Average Loss: {avg_loss:.4f}")

    # 6. Inference and Post-Processing
    print("\n=== Running Inference and Post-Processing ===")

    # Run inference on validation set
    start_logits, end_logits, ans_logits = inference_fn(
        val_loader, model, Config.DEVICE
    )

    # Verify output shapes
    assert start_logits.shape[0] == len(val_ds), "Start logits shape mismatch"
    assert end_logits.shape[0] == len(val_ds), "End logits shape mismatch"
    assert ans_logits.shape[0] == len(val_ds), "Answerability logits shape mismatch"

    # To post-process, we need the original examples (contexts)
    # We load the raw validation data used by the debug process
    # Note: In debug mode, get_processed_data creates a subset.
    # We read the full file and take the head to match the debug logic in data.py
    df_val_raw = pd.read_csv(Config.VAL_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    val_examples = df_val_raw.to_dict("records")

    # Post-process to get string predictions
    predictions = post_process_predictions(
        examples=val_examples,
        features=val_features,
        predictions=(start_logits, end_logits, ans_logits),
        n_best_size=10,
        max_answer_length=30,
    )

    assert isinstance(predictions, dict), "Predictions should be a dictionary"
    assert len(predictions) > 0, "No predictions generated"

    print("Sample Predictions:")
    for i, (uid, pred_text) in enumerate(predictions.items()):
        if i >= 3:
            break
        print(f"ID: {uid} -> Pred: '{pred_text}'")

    # 7. Metric Verification
    print("\n=== Verifying Metric Calculation ===")
    # Test Jaccard with known strings
    s1 = "machine learning model"
    s2 = "machine learning"
    score = jaccard(s1, s2)
    # Intersection: {machine, learning} (2)
    # Union: {machine, learning, model} (3)
    # Score: 2/3 ≈ 0.666
    expected_score = 2.0 / 3.0
    assert (
        abs(score - expected_score) < 1e-6
    ), f"Jaccard calculation error. Got {score}, expected {expected_score}"

    # Calculate metric on the validation subset
    print("Calculating aggregate metric on validation subset...")
    total_score = 0
    count = 0

    # Create a map of ground truths
    gt_map = {row["id"]: row["answer_text"] for row in val_examples}

    for uid, pred in predictions.items():
        if uid in gt_map:
            gt = gt_map[uid]
            # Handle NaN in ground truth if any (though dataset analysis showed none)
            if not isinstance(gt, str):
                gt = ""

            j_score = jaccard(gt, pred)
            total_score += j_score
            count += 1

    avg_jaccard = total_score / count if count > 0 else 0.0
    print(f"Validation Jaccard Score (Untrained Model): {avg_jaccard:.4f}")

    # 8. Save Submission Format (Mock)
    print("\n=== Generating Submission File ===")
    submission_df = pd.DataFrame(
        [{"id": uid, "PredictionString": pred} for uid, pred in predictions.items()]
    )
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
