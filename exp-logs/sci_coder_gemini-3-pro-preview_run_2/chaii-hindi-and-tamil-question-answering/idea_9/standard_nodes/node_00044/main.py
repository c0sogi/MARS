import os
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Import provided library components
from library.config import Config
from library.utils import set_seed, jaccard, post_process_predictions
from library.data import get_processed_data
from library.model import CustomXLMRoberta
from library.engine import get_optimizer_grouped_parameters, train_fn, inference_fn


def main():
    # 1. Setup
    set_seed(Config.SEEDS[0])
    device = torch.device(Config.DEVICE)

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print(f"Device: {device}")
    print("Initializing Data Loading...")

    # 2. Data Loading
    # Load processed datasets (Dataset objects and raw feature lists)
    train_ds, train_features = get_processed_data("train", load_cached_data=True)
    val_ds, val_features = get_processed_data("val", load_cached_data=True)

    # Load raw validation data for ground truth text
    val_df = pd.read_csv(Config.VAL_PATH)
    val_examples = val_df.to_dict("records")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Containers for ensemble logits
    ensemble_val_start_logits = []
    ensemble_val_end_logits = []
    ensemble_val_ans_logits = []

    # 3. Training Loop (Ensemble)
    print(f"Starting training with {len(Config.SEEDS)} seeds...")

    for seed in Config.SEEDS:
        print(f"\n=== Training Seed {seed} ===")
        set_seed(seed)

        # Initialize Model
        model = CustomXLMRoberta(pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        optimizer_params = get_optimizer_grouped_parameters(
            model, Config.LEARNING_RATE, Config.WEIGHT_DECAY, Config.LLRD_DECAY
        )
        optimizer = torch.optim.AdamW(optimizer_params)

        num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
        max_train_steps = Config.EPOCHS * num_update_steps_per_epoch

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(max_train_steps * Config.WARMUP_RATIO),
            num_training_steps=max_train_steps,
        )

        # Train
        for epoch in range(Config.EPOCHS):
            train_fn(train_loader, model, optimizer, device, scheduler, epoch)

        # Save Checkpoint
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Validation Inference for this seed
        print("Running validation inference for current seed...")
        start_logits, end_logits, ans_logits = inference_fn(val_loader, model, device)

        ensemble_val_start_logits.append(start_logits)
        ensemble_val_end_logits.append(end_logits)
        ensemble_val_ans_logits.append(ans_logits)

        # Cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Validation & Evaluation
    print("\n=== Aggregating Ensemble Predictions ===")

    # Average logits across seeds
    avg_val_start = np.mean(ensemble_val_start_logits, axis=0)
    avg_val_end = np.mean(ensemble_val_end_logits, axis=0)
    avg_val_ans = np.mean(ensemble_val_ans_logits, axis=0)

    # Post-process to get text
    val_preds_map = post_process_predictions(
        val_examples, val_features, (avg_val_start, avg_val_end, avg_val_ans)
    )

    # Compute Metric
    scores = []
    for item in val_examples:
        example_id = item["id"]
        gt_text = item["answer_text"]
        pred_text = val_preds_map.get(example_id, "")
        score = jaccard(gt_text, pred_text)
        scores.append(score)

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Add metrics to dataframe for analysis
    val_df["predicted_text"] = val_df["id"].map(val_preds_map)
    val_df["jaccard"] = scores
    val_df["error"] = 1.0 - val_df["jaccard"]

    # Compute lengths
    val_df["context_len"] = val_df["context"].astype(str).apply(len)
    val_df["question_len"] = val_df["question"].astype(str).apply(len)

    # Correlations
    corr_ctx = val_df["error"].corr(val_df["context_len"])
    corr_q = val_df["error"].corr(val_df["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
    print(f"Correlation (Error vs Question Length): {corr_q:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.5907916666666666

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_ds, test_features = get_processed_data("test", load_cached_data=True)
        test_df = pd.read_csv(Config.TEST_PATH)
        test_examples = test_df.to_dict("records")

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        ensemble_test_start = []
        ensemble_test_end = []
        ensemble_test_ans = []

        # Inference with each saved model
        for seed in Config.SEEDS:
            print(f"Inference on Test with Seed {seed}...")
            model = CustomXLMRoberta(pretrained=False)
            checkpoint_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)

            s, e, a = inference_fn(test_loader, model, device)
            ensemble_test_start.append(s)
            ensemble_test_end.append(e)
            ensemble_test_ans.append(a)

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Average Test Logits
        avg_test_start = np.mean(ensemble_test_start, axis=0)
        avg_test_end = np.mean(ensemble_test_end, axis=0)
        avg_test_ans = np.mean(ensemble_test_ans, axis=0)

        # Post-process
        test_preds_map = post_process_predictions(
            test_examples, test_features, (avg_test_start, avg_test_end, avg_test_ans)
        )

        # Create Submission DataFrame
        submission_data = []
        for ex in test_examples:
            ex_id = ex["id"]
            # Quote the string as per format requirement: "PredictionString"
            # However, standard CSV writers handle quoting.
            # The prompt example shows: id,PredictionString \n 8c8ee6504,"1"
            # Pandas to_csv with quoting=csv.QUOTE_NONNUMERIC or default usually works.
            # But the prompt specifically requested the string to be quoted.
            # We will rely on pandas to handle CSV escaping/quoting correctly.
            pred_str = test_preds_map.get(ex_id, "")
            submission_data.append({"id": ex_id, "PredictionString": pred_str})

        sub_df = pd.DataFrame(submission_data)

        # Save
        save_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
