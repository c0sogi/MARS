import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library import data, model, engine, utils


def main():
    # 1. Setup
    utils.seed_everything(Config.SEED)

    # 2. Data Loading
    # Load processed features (cached or computed)
    train_features, test_features = data.get_data(load_cached_data=True)

    # Load raw metadata for post-processing and evaluation
    # We need the original text contexts and answers to compute Jaccard and reconstruct predictions
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    # Combine to match the scope of train_features which includes both train and val metadata
    raw_train_data = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Dictionary to store Out-Of-Fold (OOF) predictions
    # Key: example_id, Value: predicted_text
    oof_predictions = {}

    # 3. K-Fold Training
    print(f"Starting training with {Config.N_FOLDS} folds...")

    for fold in range(Config.N_FOLDS):
        print(f"--- Fold {fold} ---")

        # Get train/val split for this fold
        train_df, val_df = data.get_folds(train_features, fold)

        # Create Datasets
        train_ds = data.QADataset(train_df, mode="train")
        val_ds = data.QADataset(val_df, mode="train")

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        net = model.XLMROBERTAForQA(pretrained=True)
        net.to(Config.DEVICE)

        # Optimizer and Scheduler
        optimizer = AdamW(
            net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
        )

        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # Training Loop
        best_val_loss = float("inf")
        best_logits = None

        for epoch in range(Config.EPOCHS):
            train_loss = engine.train_fn(
                train_loader, net, optimizer, Config.DEVICE, scheduler
            )
            val_loss, (start_logits, end_logits) = engine.eval_fn(
                val_loader, net, Config.DEVICE
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_logits = (start_logits, end_logits)
                torch.save(
                    net.state_dict(),
                    os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth"),
                )

        # Generate OOF Predictions for this fold using the best logits
        # We need raw examples corresponding to the validation features of this fold
        val_example_ids = val_df["example_id"].unique()
        val_raw_examples = raw_train_data[
            raw_train_data["id"].isin(val_example_ids)
        ].to_dict("records")
        val_features_list = val_df.to_dict("records")

        fold_preds = utils.postprocess_qa_predictions(
            examples=val_raw_examples,
            features=val_features_list,
            predictions=best_logits,
            n_best_size=Config.N_BEST_SIZE,
            max_answer_length=Config.MAX_ANSWER_LENGTH,
        )

        oof_predictions.update(fold_preds)

        # Cleanup
        del (
            net,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_ds,
            val_ds,
            best_logits,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Evaluation on Hold-out Validation Set
    print("Evaluating on hold-out validation set...")

    # We strictly evaluate on the IDs present in metadata/val.csv
    jaccard_scores = []
    analysis_records = []

    for _, row in df_val_meta.iterrows():
        eid = row["id"]
        ground_truth = row["answer_text"]

        # Get prediction
        prediction = oof_predictions.get(eid, "")

        score = utils.jaccard(ground_truth, prediction)
        jaccard_scores.append(score)

        analysis_records.append(
            {
                "jaccard": score,
                "error": 1.0 - score,
                "context_len": len(str(row["context"])),
                "question_len": len(str(row["question"])),
                "answer_len": len(str(ground_truth)),
            }
        )

    final_metric = np.mean(jaccard_scores)
    # Print full precision as requested
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    df_analysis = pd.DataFrame(analysis_records)

    corr_ctx = df_analysis["context_len"].corr(df_analysis["error"])
    corr_q = df_analysis["question_len"].corr(df_analysis["error"])
    corr_ans = df_analysis["answer_len"].corr(df_analysis["error"])

    print(f"Correlation Error vs Context Length: {corr_ctx}")
    print(f"Correlation Error vs Question Length: {corr_q}")
    print(f"Correlation Error vs Answer Length: {corr_ans}")

    # 6. Submission
    THRESHOLD = 0.2522202380952381

    if final_metric > THRESHOLD:
        print("Metric exceeds threshold. Generating submission...")

        # Load Test Data
        df_test_raw = pd.read_csv(Config.TEST_CSV)
        test_features_list = test_features.to_dict("records")

        test_ds = data.QADataset(test_features, mode="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        avg_start_logits = None
        avg_end_logits = None

        # Ensemble Inference
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth")
            net = model.XLMROBERTAForQA(pretrained=False)
            net.load_state_dict(torch.load(model_path))
            net.to(Config.DEVICE)

            _, (start_logits, end_logits) = engine.eval_fn(
                test_loader, net, Config.DEVICE
            )

            if avg_start_logits is None:
                avg_start_logits = start_logits
                avg_end_logits = end_logits
            else:
                avg_start_logits += start_logits
                avg_end_logits += end_logits

            del net
            torch.cuda.empty_cache()
            gc.collect()

        # Average logits
        avg_start_logits /= Config.N_FOLDS
        avg_end_logits /= Config.N_FOLDS

        # Post-process
        test_preds = utils.postprocess_qa_predictions(
            examples=df_test_raw.to_dict("records"),
            features=test_features_list,
            predictions=(avg_start_logits, avg_end_logits),
            n_best_size=Config.N_BEST_SIZE,
            max_answer_length=Config.MAX_ANSWER_LENGTH,
        )

        # Format Submission
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        sub_df = pd.DataFrame(
            list(test_preds.items()), columns=["id", "PredictionString"]
        )
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation metric {final_metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
