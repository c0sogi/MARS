import pandas as pd
import numpy as np
import torch
import os
import sys
import shutil

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import (
    get_tokenizer,
    get_train_dataloader,
    process_data_to_features,
    QADataset,
)
from library.model import CustomXLMRoberta
from library.engine import get_optimizer, get_scheduler, train_loop, predict_fn
from library.inference import post_process_predictions, run_inference


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config to ensure we don't train on validation data (Leakage Prevention)
    Config.use_full_train_data = False

    # Clear training cache to ensure the new data split is respected
    if os.path.exists(Config.train_features_file):
        try:
            os.remove(Config.train_features_file)
            print(
                f"Cleared cache at {Config.train_features_file} to enforce data split."
            )
        except OSError as e:
            print(f"Error removing cache: {e}")

    device = Config.device
    tokenizer = get_tokenizer()

    # =========================================================================
    # 2. Training Loop (Adversarial Seed Ensemble)
    # =========================================================================
    print(f"Starting training with {Config.epochs} epochs per seed...")

    # Load training data (load_cached_data=False forces regeneration with strict split)
    train_loader = get_train_dataloader(tokenizer, load_cached_data=False)

    for seed in Config.seeds:
        print(f"\n--- Training Seed {seed} ---")
        seed_everything(seed)

        # Initialize Model
        model = CustomXLMRoberta()
        model.to(device)

        # Optimizer & Scheduler
        optimizer = get_optimizer(model)
        num_training_steps = len(train_loader) * Config.epochs
        scheduler = get_scheduler(optimizer, num_training_steps)

        # Save Path
        save_path = os.path.join(Config.output_dir, f"model_seed_{seed}.pth")

        # Execute Training
        # We pass None for val_dataloader to train_loop because we handle validation
        # separately after the ensemble is fully trained.
        train_loop(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=Config.epochs,
            save_path=save_path,
        )

        # Cleanup to free GPU memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("\n--- Running Validation on Hold-out Set ---")

    # Load Validation Metadata
    val_df = pd.read_csv(Config.val_meta_path)

    # Generate Validation Features (reuse process_data_to_features with is_test=True for sliding windows)
    # We treat validation as "test" during inference phase to get offset mappings
    val_features = process_data_to_features(val_df, tokenizer, is_test=True)

    # Ensure list types for Parquet/Dataset compatibility
    val_features["input_ids"] = val_features["input_ids"].apply(list)
    val_features["attention_mask"] = val_features["attention_mask"].apply(list)

    val_dataset = QADataset(val_features, is_test=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Ensemble Inference
    avg_start_logits = None
    avg_end_logits = None
    avg_rel_logits = None
    models_count = 0

    for seed in Config.seeds:
        model_path = os.path.join(Config.output_dir, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found.")
            continue

        print(f"Predicting with seed {seed}...")
        model = CustomXLMRoberta()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        s_log, e_log, r_log = predict_fn(val_loader, model, device)

        if avg_start_logits is None:
            avg_start_logits = s_log
            avg_end_logits = e_log
            avg_rel_logits = r_log
        else:
            avg_start_logits += s_log
            avg_end_logits += e_log
            avg_rel_logits += r_log

        models_count += 1
        del model
        torch.cuda.empty_cache()

    if models_count > 0:
        avg_start_logits /= models_count
        avg_end_logits /= models_count
        avg_rel_logits /= models_count

        # Post-process predictions
        # Hack: library.inference.post_process_predictions reads Config.test_meta_path
        # We temporarily point it to val_meta_path so it can map IDs to Contexts correctly
        original_test_path = Config.test_meta_path
        Config.test_meta_path = Config.val_meta_path

        try:
            preds_map = post_process_predictions(
                val_features, avg_start_logits, avg_end_logits, avg_rel_logits
            )
        finally:
            # Restore original path
            Config.test_meta_path = original_test_path

        # Compute Metric
        scores = []
        val_df["prediction"] = val_df["id"].map(preds_map).fillna("")

        for _, row in val_df.iterrows():
            gt = str(row["answer_text"]) if not pd.isna(row["answer_text"]) else ""
            pred = str(row["prediction"])
            scores.append(jaccard(gt, pred))

        val_metric = np.mean(scores)
        print(f"Final Validation Metric: {val_metric}")

        # =========================================================================
        # 4. Failure Analysis
        # =========================================================================
        print("\n--- Failure Analysis ---")
        val_df["jaccard"] = scores
        val_df["error"] = 1.0 - val_df["jaccard"]

        # Calculate lengths
        val_df["context_len"] = val_df["context"].apply(len)
        val_df["question_len"] = val_df["question"].apply(len)
        val_df["answer_len"] = val_df["answer_text"].apply(lambda x: len(str(x)))

        # Correlation
        correlations = val_df[
            ["error", "context_len", "question_len", "answer_len"]
        ].corr()["error"]
        print("Correlation between Error (1-Jaccard) and Features:")
        print(correlations)

        # =========================================================================
        # 5. Submission
        # =========================================================================
        if val_metric > 0.616:
            print("\nMetric threshold (0.616) passed. Generating submission...")
            # run_inference uses the models saved in Config.output_dir and Config.test_meta_path
            run_inference()
        else:
            print(
                f"\nMetric {val_metric} did not pass threshold 0.616. Skipping submission."
            )

    else:
        print("No models were trained successfully. Cannot validate.")


if __name__ == "__main__":
    main()
