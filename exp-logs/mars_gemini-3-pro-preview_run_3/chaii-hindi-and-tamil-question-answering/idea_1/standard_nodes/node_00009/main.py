import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, logging

# Import provided library components
from library.config import Config, set_seed
from library.utils import jaccard
from library.model import QATokenClassifier
from library.data_loader import prepare_data
from library.trainer import Trainer

# Suppress excessive transformer warnings
logging.set_verbosity_error()


def main():
    # 1. Configuration and Setup
    # Increase epochs to allow better convergence
    Config.EPOCHS = 10
    Config.setup()

    print("Initializing Fast Baseline Run...")

    # 2. Data Loading
    # Load processed datasets (using cache if available)
    # Disable cache to ensure validation data alignment fix is applied
    train_dataset, val_dataset, test_dataset = prepare_data(load_cached_data=False)

    # Load raw metadata for alignment and analysis
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Seed Ensembling Loop (Cite solution_lesson_node_00008)
    seeds = [42, 43, 44]
    model_paths = []

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Placeholder for trainer to be used in inference
    trainer = None

    for seed in seeds:
        print(f"\n=== Starting Training for Seed {seed} ===")
        set_seed(seed)

        # Re-initialize model for each seed
        model = QATokenClassifier(Config.MODEL_NAME)
        model.to(Config.DEVICE)

        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_training_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        trainer = Trainer(model, tokenizer, Config.DEVICE)

        best_val_score = -1.0
        seed_save_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pt")

        for epoch in range(Config.EPOCHS):
            train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
            val_score = trainer.validate(val_loader, df_val)

            print(
                f"Seed {seed} | Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss:.4f} | Val: {val_score:.4f}"
            )

            if val_score > best_val_score:
                best_val_score = val_score
                trainer.save_model(seed_save_path)
                print(f"  -> Saved Best (Score: {best_val_score:.4f})")

        model_paths.append(seed_save_path)

    # 4. Ensemble Evaluation
    print("\n=== Ensemble Evaluation ===")

    # Accumulate probabilities
    avg_probs = None
    ref_input_ids = None
    ref_sample_idxs = None

    for path in model_paths:
        print(f"Loading {path}...")
        trainer.load_model(path)
        probs, input_ids, sample_idxs = trainer.get_probs(val_loader)

        if avg_probs is None:
            avg_probs = probs
            ref_input_ids = input_ids
            ref_sample_idxs = sample_idxs
        else:
            avg_probs += probs

    # Compute average
    avg_probs /= len(model_paths)

    # Decode using Greedy First-Match (Cite solution_lesson_node_00007)
    all_preds = trainer.decode_from_probs(avg_probs, ref_input_ids, ref_sample_idxs)

    # Compute Final Metric
    scores = []
    val_ids = []
    val_preds = []

    for idx, row in df_val.iterrows():
        uid = row["id"]
        gt = row["answer_text"]

        if idx in all_preds:
            pred = all_preds[idx][1]
        else:
            pred = ""

        val_ids.append(uid)
        val_preds.append(pred)
        scores.append(jaccard(gt, pred))

    final_metric = np.mean(scores)
    print(f"Final Ensemble Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = df_val.copy()
    df_analysis["jaccard"] = scores
    df_analysis["error"] = 1.0 - df_analysis["jaccard"]

    # Feature Engineering for Analysis
    df_analysis["context_len"] = df_analysis["context"].astype(str).apply(len)
    df_analysis["question_len"] = df_analysis["question"].astype(str).apply(len)
    df_analysis["is_tamil"] = (df_analysis["language"] == "tamil").astype(int)

    # Calculate Correlations
    correlations = df_analysis[
        ["error", "context_len", "question_len", "is_tamil"]
    ].corr()["error"]

    print("Correlation between Error (1-Jaccard) and Input Features:")
    print(f"  Context Length:  {correlations['context_len']:.4f}")
    print(f"  Question Length: {correlations['question_len']:.4f}")
    print(f"  Language (Tamil): {correlations['is_tamil']:.4f}")

    # 6. Submission Generation
    if final_metric > 0.3011529653320698:
        print("\nGenerating submission for test set...")

        # Ensemble Inference on Test
        test_avg_probs = None
        test_input_ids = None
        test_sample_idxs = None

        for path in model_paths:
            trainer.load_model(path)
            probs, input_ids, sample_idxs = trainer.get_probs(test_loader)

            if test_avg_probs is None:
                test_avg_probs = probs
                test_input_ids = input_ids
                test_sample_idxs = sample_idxs
            else:
                test_avg_probs += probs

        test_avg_probs /= len(model_paths)

        test_preds_map = trainer.decode_from_probs(
            test_avg_probs, test_input_ids, test_sample_idxs
        )

        test_ids_out = []
        test_preds_out = []

        for idx, row in df_test.iterrows():
            test_ids_out.append(row["id"])
            if idx in test_preds_map:
                test_preds_out.append(test_preds_map[idx][1])
            else:
                test_preds_out.append("")

        submission_df = pd.DataFrame(
            {"id": test_ids_out, "PredictionString": test_preds_out}
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nFinal metric {final_metric:.4f} is not higher than threshold. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
