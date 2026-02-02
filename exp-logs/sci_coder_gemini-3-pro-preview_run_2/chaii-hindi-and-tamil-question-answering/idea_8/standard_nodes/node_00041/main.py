import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import prepare_train_features, prepare_test_features, QADataset
from library.model import CustomXLMRoberta
from library.engine import train_fn, get_optimizer_grouped_parameters
from library.inference import inference_fn, get_best_span


def run_training():
    """
    Orchestrates the training process for the ensemble.
    Trains a model for each seed defined in Config.
    """
    print("=== Starting Training Phase ===")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Load Training Data
    print(f"Loading training data from {Config.train_path}")
    if not os.path.exists(Config.train_path):
        raise FileNotFoundError(f"Train file not found at {Config.train_path}")
    df_train = pd.read_csv(Config.train_path)

    # Prepare Features
    # caching is enabled to speed up re-runs or debugging
    train_features = prepare_train_features(df_train, load_cached_data=True)

    # Create Dataset and DataLoader
    train_dataset = QADataset(train_features, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Training Loop for each seed
    for seed in Config.seeds:
        print(f"\n--- Training Seed {seed} ---")
        seed_everything(seed)

        # Initialize Model
        model = CustomXLMRoberta()
        model.to(Config.device)

        # Initialize Optimizer with LLRD
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, Config)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=Config.learning_rate, eps=1e-6
        )

        # Initialize Scheduler
        # Calculate total training steps
        num_update_steps_per_epoch = (
            len(train_features)
            // Config.train_batch_size
            // Config.accumulate_grad_batches
        )
        num_train_steps = num_update_steps_per_epoch * Config.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Train for specified epochs
        for epoch in range(Config.epochs):
            print(f"Epoch {epoch + 1}/{Config.epochs}")
            train_fn(train_loader, model, optimizer, Config.device, scheduler, Config)

        # Save the model checkpoint for this seed
        save_path = os.path.join(Config.working_dir, f"best_model_seed_{seed}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Clean up to free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()


def run_validation():
    """
    Performs validation using the ensemble of trained models.
    Calculates the Jaccard metric and performs failure analysis.
    """
    print("\n=== Starting Validation Phase ===")

    # Explicitly remove test cache to prevent collision with validation data
    # prepare_test_features writes to a fixed filename 'cached_test_features.parquet'
    cache_path = os.path.join(Config.working_dir, "cached_test_features.parquet")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # Load Validation Data
    print(f"Loading validation data from {Config.val_path}")
    if not os.path.exists(Config.val_path):
        raise FileNotFoundError(f"Validation file not found at {Config.val_path}")
    df_val = pd.read_csv(Config.val_path)

    # Prepare Features
    # We use prepare_test_features to get offset mappings and context for text extraction.
    # We disable loading cached data to force generation from df_val.
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    val_features, _ = prepare_test_features(
        df_val, tokenizer=tokenizer, load_cached_data=False
    )

    val_dataset = QADataset(val_features, mode="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize accumulators for Ensemble Inference
    num_samples = len(val_features)
    seq_len = Config.max_len

    avg_start_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_ans_probs = np.zeros((num_samples,), dtype=np.float32)

    models_used = 0

    # Iterate through seeds and aggregate predictions
    for seed in Config.seeds:
        model_path = os.path.join(Config.working_dir, f"best_model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Inference with seed {seed} model...")
        model = CustomXLMRoberta()
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        models_used += 1

        batch_start_preds = []
        batch_end_preds = []
        batch_ans_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)

                s, e, a = model(input_ids, attention_mask)

                batch_start_preds.append(s.cpu().numpy())
                batch_end_preds.append(e.cpu().numpy())
                batch_ans_preds.append(torch.sigmoid(a).squeeze(-1).cpu().numpy())

        avg_start_logits += np.concatenate(batch_start_preds, axis=0)
        avg_end_logits += np.concatenate(batch_end_preds, axis=0)
        avg_ans_probs += np.concatenate(batch_ans_preds, axis=0)

        del model
        torch.cuda.empty_cache()

    if models_used == 0:
        raise RuntimeError("No models available for validation.")

    # Average logits
    avg_start_logits /= models_used
    avg_end_logits /= models_used
    avg_ans_probs /= models_used

    # Extract Best Spans
    print("Calculating metrics...")

    # Map predictions back to original examples using example_id
    preds_map = {}  # id -> (score, prediction_string)

    for idx, feature in enumerate(val_features):
        eid = feature["example_id"]

        score, text = get_best_span(
            avg_start_logits[idx],
            avg_end_logits[idx],
            avg_ans_probs[idx],
            feature["sequence_ids"],
            feature["offset_mapping"],
            feature["context"],
        )

        # Keep the prediction with the highest score for this example_id (handling sliding windows)
        if eid not in preds_map or score > preds_map[eid][0]:
            preds_map[eid] = (score, text)

    # Calculate Jaccard Metric
    jaccard_scores = []
    error_magnitudes = []
    context_lengths = []
    question_lengths = []

    for _, row in df_val.iterrows():
        eid = row["id"]
        gt_text = str(row["answer_text"])

        # Get prediction, default to empty string if missing
        if eid in preds_map:
            pred_text = preds_map[eid][1]
        else:
            pred_text = ""

        score = jaccard(gt_text, pred_text)
        jaccard_scores.append(score)

        # Collect data for failure analysis
        error_magnitudes.append(1.0 - score)
        context_lengths.append(len(str(row["context"])))
        question_lengths.append(len(str(row["question"])))

    final_metric = np.mean(jaccard_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(
        {
            "error": error_magnitudes,
            "context_len": context_lengths,
            "question_len": question_lengths,
        }
    )

    corr_ctx = df_analysis["error"].corr(df_analysis["context_len"])
    corr_q = df_analysis["error"].corr(df_analysis["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
    print(f"Correlation (Error vs Question Length): {corr_q:.4f}")

    # Cleanup cache again
    if os.path.exists(cache_path):
        os.remove(cache_path)

    return final_metric


def main():
    # 1. Train models
    run_training()

    # 2. Validate and Analyze
    metric = run_validation()

    # 3. Submit if threshold met
    threshold = 0.5907916666666666
    if metric > threshold:
        print(
            f"\nMetric ({metric}) > Threshold ({threshold}). Generating submission..."
        )

        # Clear cache to ensure test features are generated correctly
        cache_path = os.path.join(Config.working_dir, "cached_test_features.parquet")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        # Run inference (generates submission.csv in Config.working_dir)
        inference_fn()

        # Move submission to required directory
        target_dir = "./submission"
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "submission.csv")

        source_path = Config.submission_path
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Submission copied to {target_path}")
        else:
            print(f"Error: Source submission file {source_path} not found.")

    else:
        print(f"\nMetric ({metric}) <= Threshold ({threshold}). Skipping submission.")


if __name__ == "__main__":
    main()
