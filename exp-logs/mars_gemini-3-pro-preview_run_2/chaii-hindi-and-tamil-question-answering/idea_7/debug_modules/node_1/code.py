import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import get_data, QADataset
from library.model import XLMRobertaForMultiTaskQA
from library.engine import train_one_epoch, validate
from library.postprocessing import postprocess_predictions, save_submission


def run_demo():
    print("=== Starting QA Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config settings for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # We use 'xlm-roberta-base' instead of 'large' for faster download and execution
    # in this demonstration script.
    DEMO_MODEL_NAME = "xlm-roberta-base"

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Configuration: Debug={Config.DEBUG}, Model={DEMO_MODEL_NAME}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[Data] Loading and processing data...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEMO_MODEL_NAME)

    # Load Training Data
    # load_cached_data=False ensures we process the debug subset fresh
    train_features = get_data(
        tokenizer=tokenizer, load_cached_data=False, split="train"
    )
    print(f"Train features shape: {train_features.shape}")

    # Load Validation Data
    val_features = get_data(tokenizer=tokenizer, load_cached_data=False, split="val")
    print(f"Validation features shape: {val_features.shape}")

    # Create Datasets
    # is_test=False ensures we get labels (start_positions, etc.) for training/val loss
    train_dataset = QADataset(train_features, is_test=False)
    val_dataset = QADataset(val_features, is_test=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Verify Data Loading
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "start_positions" in sample_batch
    print("[Data] Batch verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[Model] Initializing model...")
    device = Config.DEVICE
    # Initialize model with the base config for speed
    model = XLMRobertaForMultiTaskQA(model_name=DEMO_MODEL_NAME)
    model.to(device)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\n[Training] Starting training loop...")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Train for 1 epoch
    avg_train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loader=train_loader,
        device=device,
        epoch=0,
    )
    print(f"[Training] Epoch 0 Loss: {avg_train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Validation & Inference
    # -------------------------------------------------------------------------
    print("\n[Validation] Running inference on validation set...")

    # Run validation to get logits and loss
    val_outputs = validate(model, val_loader, device)

    # Verify outputs structure
    assert "start_logits" in val_outputs
    assert "end_logits" in val_outputs
    assert "answerability_logits" in val_outputs
    assert val_outputs["loss"] is not None
    print(f"[Validation] Loss: {val_outputs['loss']:.4f}")

    # -------------------------------------------------------------------------
    # 6. Post-processing & Metrics
    # -------------------------------------------------------------------------
    print("\n[Post-processing] Generating text predictions...")

    # Load raw validation metadata to get original context text
    # Since we used DEBUG mode in get_data, we must slice the raw data similarly
    # to align with the processed features.
    raw_val_df = pd.read_csv(Config.VAL_META)
    if Config.DEBUG:
        raw_val_df = raw_val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Generate text predictions from logits
    submission_df = postprocess_predictions(
        examples=raw_val_df,
        features=val_features,
        raw_predictions=val_outputs,
        n_best_size=10,
        max_answer_length=30,
    )

    print("[Post-processing] Predictions generated.")

    # Calculate Jaccard Score
    print("[Metrics] Calculating Jaccard Score...")
    merged_df = pd.merge(raw_val_df, submission_df, on="id")

    jaccard_scores = []
    for idx, row in merged_df.iterrows():
        ground_truth = str(row["answer_text"]) if pd.notna(row["answer_text"]) else ""
        prediction = (
            str(row["PredictionString"]) if pd.notna(row["PredictionString"]) else ""
        )
        score = jaccard(ground_truth, prediction)
        jaccard_scores.append(score)

    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    print(f"Mean Jaccard Score: {mean_jaccard:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(submission_df, output_path=output_path)

    assert os.path.exists(output_path), "Submission file was not created."
    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
