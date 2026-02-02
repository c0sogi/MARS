import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_optimizer_grouped_parameters
from library.data import prepare_data, process_examples, QADataset
from library.model import CustomXLMRoberta
from library.engine import train_fn
from library.inference import run_inference, get_best_span

# =============================================================================
# Configuration Overrides for Fast Baseline & Validation
# =============================================================================
# We override the default Config to ensure we have a hold-out validation set
# and to speed up the training for this baseline run.
Config.MERGE_TRAIN_VAL = False  # Split train/val to compute metric
Config.EPOCHS = 3  # Reduced epochs for fast baseline
Config.SEEDS = [42]  # Single seed for baseline
Config.NUM_WORKERS = 2  # Conservative worker count


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))


def main():
    # 1. Setup
    seed_everything(42)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Preparation
    print("Preparing Training Data...")
    # prepare_data will now only load train.csv (metadata) because MERGE_TRAIN_VAL is False
    # We disable cache loading to ensure we don't load a previously merged dataset
    train_dataset, _ = prepare_data(load_cached_data=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Preparing Validation Data...")
    # Manually process validation data to get offset mappings (required for inference/metric)
    val_df = pd.read_csv(Config.VAL_META)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process with is_train=False to get offset_mapping and example_idx
    val_features = process_examples(
        val_df,
        tokenizer,
        is_train=False,
        max_length=Config.MAX_LENGTH,
        doc_stride=Config.DOC_STRIDE,
    )

    val_dataset = QADataset(val_features, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = CustomXLMRoberta(Config.MODEL_NAME)
    model.to(device)

    # Optimization setup
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, Config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 4. Training Loop
    print(f"Starting Training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Training Loss: {avg_loss:.4f}")

    # Save the model (required for run_inference later)
    save_path = os.path.join(Config.OUTPUT_DIR, "model_seed_42.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 5. Validation Inference & Metric Calculation
    print("Running Inference on Validation Set...")
    model.eval()

    # Store best result per example: {example_idx: (score, answer_text)}
    results = {}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offset_mapping = batch["offset_mapping"].numpy()
            example_indices = batch["example_idx"].numpy()

            start_logits, end_logits, rel_logits = model(input_ids, attention_mask)

            # Move to CPU
            s = start_logits.cpu().numpy()
            e = end_logits.cpu().numpy()
            r = rel_logits.cpu().numpy()

            for i in range(len(example_indices)):
                ex_idx = example_indices[i]
                offsets = offsets = offset_mapping[i]

                # Retrieve context
                context = val_df.iloc[ex_idx]["context"]

                # Extract best span
                score, text = get_best_span(s[i], e[i], r[i][0], offsets, context)

                # Update best result for this example ID
                if ex_idx not in results or score > results[ex_idx][0]:
                    results[ex_idx] = (score, text)

    # Calculate Jaccard Metric
    total_jaccard = 0.0
    count = 0

    # Data for failure analysis
    analysis_records = []

    for idx, row in val_df.iterrows():
        pred_text = ""
        if idx in results:
            pred_text = results[idx][1]

        gt_text = str(row["answer_text"])
        score = jaccard(gt_text, pred_text)

        total_jaccard += score
        count += 1

        # Collect metadata for failure analysis
        analysis_records.append(
            {
                "jaccard": score,
                "error": 1.0 - score,
                "context_len": len(str(row["context"])),
                "question_len": len(str(row["question"])),
                "answer_len": len(gt_text),
            }
        )

    final_metric = total_jaccard / count if count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    analysis_df = pd.DataFrame(analysis_records)

    # Calculate correlation between error and features
    # We select numeric columns for correlation
    corr_cols = ["error", "context_len", "question_len", "answer_len"]
    correlations = analysis_df[corr_cols].corr()["error"]

    print("Correlation with Error Magnitude (1 - Jaccard):")
    print(correlations.drop("error"))  # Drop self-correlation

    # 7. Submission Generation
    THRESHOLD = 0.60025
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # run_inference loads the model we just saved from Config.OUTPUT_DIR
        # We disable cache loading to ensure it processes the test set cleanly
        run_inference(load_cached_data=False)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
