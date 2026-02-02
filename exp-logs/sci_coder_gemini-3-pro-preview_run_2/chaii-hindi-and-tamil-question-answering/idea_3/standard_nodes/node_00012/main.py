import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold

# Import from provided library
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import process_data, QADataset, get_data
from library.model import get_model, get_tokenizer
from library.engine import train_one_epoch, get_predictions
from library.inference import generate_submission


def run_training():
    seed_everything(Config.SEED)

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    print("Loading training data from metadata...")
    df_train_full = pd.read_csv(Config.TRAIN_CSV)

    # Initialize Tokenizer
    tokenizer = get_tokenizer()

    # Prepare Group K-Fold
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    folds = list(gkf.split(df_train_full, groups=df_train_full["context"]))

    device = Config.DEVICE

    print(f"Starting training with {Config.N_FOLDS} folds on device {device}...")

    for fold, (train_idx, _) in enumerate(folds):
        print(f"\n=== Fold {fold} ===")

        # Subset data
        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)

        # Limit samples for fast baseline if needed, but dataset is small so we use full fold
        # Process features
        print("Processing fold data...")
        train_features = process_data(df_train_fold, tokenizer, has_labels=True)

        train_dataset = QADataset(train_features)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = get_model()
        model.to(device)

        # Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_training_steps = len(train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training Loop
        best_loss = float("inf")

        for epoch in range(Config.EPOCHS):
            print(f"Fold {fold} | Epoch {epoch + 1}/{Config.EPOCHS}")
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch
            )

        # Save model for this fold
        save_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved model for fold {fold} to {save_path}")

        # Cleanup
        del model, optimizer, scheduler, train_loader, train_dataset, train_features
        torch.cuda.empty_cache()


def run_validation():
    print("\n=== Running Validation on Hold-out Set ===")
    device = Config.DEVICE
    tokenizer = get_tokenizer()

    # Load validation data
    df_val = pd.read_csv(Config.VAL_CSV)
    val_features = process_data(df_val, tokenizer, has_labels=True)

    val_dataset = QADataset(val_features)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference
    avg_start_logits = None
    avg_end_logits = None
    models_found = 0

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            continue

        print(f"Inference with fold {fold} model...")
        model = get_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        start_logits, end_logits = get_predictions(model, val_loader, device)

        if avg_start_logits is None:
            avg_start_logits = start_logits
            avg_end_logits = end_logits
        else:
            avg_start_logits += start_logits
            avg_end_logits += end_logits

        models_found += 1
        del model
        torch.cuda.empty_cache()

    if models_found > 0:
        avg_start_logits /= models_found
        avg_end_logits /= models_found
    else:
        print("No models found for validation!")
        return 0.0

    # Decode and Evaluate
    print("Decoding validation predictions...")

    # Map example_id to ground truth
    gt_map = dict(zip(df_val["id"], df_val["answer_text"]))
    context_map = dict(zip(df_val["id"], df_val["context"]))

    # Store predictions
    preds_map = {}  # id -> (score, text)

    example_ids = val_features["example_id"].values
    offset_mapping = val_features["offset_mapping"].values
    sequence_ids = val_features["sequence_ids"].values

    for i in range(len(val_features)):
        eid = example_ids[i]
        offsets = offset_mapping[i]
        seq_ids = sequence_ids[i]
        context_text = context_map.get(eid, "")

        # Valid context indices
        ctx_indices = [idx for idx, s in enumerate(seq_ids) if s == 1]
        if not ctx_indices:
            continue

        min_idx, max_idx = ctx_indices[0], ctx_indices[-1]

        start_log = avg_start_logits[i]
        end_log = avg_end_logits[i]

        start_candidates = np.argsort(start_log)[-Config.N_BEST_SIZE :]
        end_candidates = np.argsort(end_log)[-Config.N_BEST_SIZE :]

        best_score = -float("inf")
        best_ans = ""

        for s_idx in start_candidates:
            if s_idx < min_idx or s_idx > max_idx:
                continue
            for e_idx in end_candidates:
                if e_idx < min_idx or e_idx > max_idx:
                    continue
                if e_idx < s_idx:
                    continue
                if e_idx - s_idx + 1 > Config.MAX_ANSWER_LENGTH:
                    continue

                score = start_log[s_idx] + end_log[e_idx]
                if score > best_score:
                    best_score = score
                    try:
                        c_start = offsets[s_idx][0]
                        c_end = offsets[e_idx][1]
                        best_ans = context_text[c_start:c_end]
                    except:
                        continue

        if eid not in preds_map or best_score > preds_map[eid][0]:
            preds_map[eid] = (best_score, best_ans)

    # Calculate Metric
    scores = []
    errors = []

    # For failure analysis
    ctx_lens = []
    q_lens = []

    for _, row in df_val.iterrows():
        eid = row["id"]
        gt = row["answer_text"]
        pred = preds_map.get(eid, (0, ""))[1]

        score = jaccard(gt, pred)
        scores.append(score)
        errors.append(1.0 - score)

        ctx_lens.append(len(str(row["context"])))
        q_lens.append(len(str(row["question"])))

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(errors) > 1:
        corr_ctx = np.corrcoef(errors, ctx_lens)[0, 1]
        corr_q = np.corrcoef(errors, q_lens)[0, 1]
        print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
        print(f"Correlation (Error vs Question Length): {corr_q:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    return final_metric


def main():
    # 1. Train
    run_training()

    # 2. Validate
    metric = run_validation()

    # 3. Submit
    threshold = 0.2522202380952381
    if metric > threshold:
        print(f"\nMetric {metric} > {threshold}. Generating submission...")
        generate_submission()

        # Move submission to required path
        src = Config.SUBMISSION_CSV
        dst = "./submission/submission.csv"
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Submission copied to {dst}")
        else:
            print(f"Error: Source submission file {src} not found.")
    else:
        print(f"\nMetric {metric} <= {threshold}. Skipping submission generation.")


if __name__ == "__main__":
    main()
