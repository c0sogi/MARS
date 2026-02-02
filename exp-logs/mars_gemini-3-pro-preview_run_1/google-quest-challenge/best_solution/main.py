import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from library.trainer import Trainer
from library.dataset import StackExchangeDataset, collate_fn
from library.model import PartitionedPoolingDualEncoder
from library.utils import seed_everything, compute_spearman_metric


def main():
    # 1. Setup
    seed_everything(42)
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_NAME = "roberta-base"
    THRESHOLD = 0.4118214482019393

    # 2. Training
    # Initialize trainer with phantom scheduling parameters
    # Total epochs 7, but stop at 3 to optimize the learning rate decay profile
    print("Initializing Trainer...")
    trainer = Trainer(
        model_name=MODEL_NAME,
        epochs=7,
        stop_epoch=3,
        batch_size=8,
        accum_steps=2,
        lr_backbone=2e-5,
        lr_head=1e-3,
        working_dir=WORKING_DIR,
        debug=False,
    )

    # Execute training (this will also generate a submission by default)
    trainer.train()

    # 3. Evaluation & Metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best model for validation inference
    print("Loading best model for validation...")
    model = PartitionedPoolingDualEncoder(model_name=MODEL_NAME)
    model.to(device)

    checkpoint_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Error: Best model checkpoint not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Validation Inference
    # We load the full validation set from metadata
    val_dataset = StackExchangeDataset(
        split="val", tokenizer_name=MODEL_NAME, cache_dir=WORKING_DIR, debug=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    preds = []
    targets = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            title_mask = batch["title_mask"].to(device)
            body_mask = batch["body_mask"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids_q,
                attention_mask_q,
                title_mask,
                body_mask,
                input_ids_a,
                attention_mask_a,
            )
            probs = torch.sigmoid(logits)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Compute Metric
    final_metric = compute_spearman_metric(targets, preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    # Compute MAE per sample (mean across the 30 targets)
    mae_per_sample = np.mean(np.abs(targets - preds), axis=1)

    # Load metadata to get features
    val_df = pd.read_csv("./metadata/val.csv")

    # Ensure alignment (truncate if necessary, though sizes should match)
    if len(val_df) != len(mae_per_sample):
        print(
            f"Warning: Mismatch in validation set size. Metadata: {len(val_df)}, Preds: {len(mae_per_sample)}"
        )
        min_len = min(len(val_df), len(mae_per_sample))
        val_df = val_df.iloc[:min_len]
        mae_per_sample = mae_per_sample[:min_len]

    # Calculate features
    val_df["q_body_len"] = val_df["question_body"].fillna("").astype(str).str.len()
    val_df["ans_len"] = val_df["answer"].fillna("").astype(str).str.len()

    # Correlations
    corr_q, _ = spearmanr(mae_per_sample, val_df["q_body_len"])
    corr_a, _ = spearmanr(mae_per_sample, val_df["ans_len"])

    print(f"Correlation (Error vs Question Body Length): {corr_q:.4f}")
    print(f"Correlation (Error vs Answer Length): {corr_a:.4f}")

    # 5. Submission Logic
    # Strict compliance: Generate if and only if metric > threshold
    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Keeping submission.")
        # Ensure submission exists (Trainer generates it, but we verify)
        if not os.path.exists(SUBMISSION_PATH):
            print("Submission file missing, regenerating...")
            trainer.generate_submission(model)
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Discarding submission.")
        if os.path.exists(SUBMISSION_PATH):
            os.remove(SUBMISSION_PATH)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
