import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import collections

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_metric, jaccard
from library.data_loader import prepare_train_features, prepare_test_features, QADataset
from library.model import MuRILForQA
from library.trainer import QATrainer
from library.predictor import Predictor


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading metadata...")
    train_df = pd.read_csv(Config.META_TRAIN_PATH)
    val_df = pd.read_csv(Config.META_VAL_PATH)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # 4. Group K-Fold Training
    # We split metadata/train.csv into K folds
    gkf = GroupKFold(n_splits=Config.NUM_FOLDS)

    print(f"Starting Group K-Fold Training (K={Config.NUM_FOLDS})...")

    for fold, (train_idx, inner_val_idx) in enumerate(
        gkf.split(train_df, groups=train_df["context"])
    ):
        print(f"\n=== Fold {fold} ===")

        # Split data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        # We use the inner split for monitoring, though the main validation happens on the hold-out set later
        fold_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)

        # Prepare Features
        print("Preparing features...")
        train_features = prepare_train_features(fold_train_df, tokenizer)
        val_features = prepare_train_features(fold_val_df, tokenizer)

        # Create DataLoaders
        train_dataset = QADataset(train_features, mode="train")
        val_dataset = QADataset(val_features, mode="train")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        model = MuRILForQA()
        model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
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

        # Trainer
        trainer = QATrainer(model, tokenizer, device, optimizer, scheduler)

        # Train Loop
        best_fold_jaccard = 0.0
        best_model_state = None

        for epoch in range(Config.EPOCHS):
            # Train
            avg_loss = trainer.train_epoch(train_loader, epoch + 1)

            # Eval (on inner fold validation)
            val_jaccard, val_loss = trainer.eval_epoch(val_loader, fold_val_df)

            print(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Jaccard={val_jaccard:.4f}"
            )

            # Save best state for this fold
            if val_jaccard >= best_fold_jaccard:
                best_fold_jaccard = val_jaccard
                best_model_state = model.state_dict()

        # Save fold model
        if best_model_state is None:
            best_model_state = model.state_dict()  # Fallback if no improvement

        save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        torch.save(best_model_state, save_path)
        print(f"Saved best model for fold {fold} to {save_path}")

        # Cleanup
        del model, optimizer, scheduler, trainer, train_loader, val_loader
        torch.cuda.empty_cache()

    # 5. Hold-out Validation (Ensemble Inference)
    print("\n=== Running Ensemble Validation on Hold-out Set ===")

    # Prepare validation features (inference mode logic to keep offsets)
    val_features_inf = prepare_test_features(val_df, tokenizer)
    val_dataset_inf = QADataset(val_features_inf, mode="test")
    val_loader_inf = DataLoader(
        val_dataset_inf,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Aggregate Logits
    num_features = len(val_features_inf)
    agg_start_logits = np.zeros((num_features, Config.MAX_LENGTH), dtype=np.float32)
    agg_end_logits = np.zeros((num_features, Config.MAX_LENGTH), dtype=np.float32)

    predictor = Predictor()
    models_loaded = 0

    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            continue

        model = MuRILForQA()
        model.load_state_dict(torch.load(model_path, map_location=device))

        s_logits, e_logits = predictor.predict_fn(model, val_loader_inf)

        agg_start_logits += s_logits
        agg_end_logits += e_logits
        models_loaded += 1

        del model
        torch.cuda.empty_cache()

    if models_loaded > 0:
        agg_start_logits /= models_loaded
        agg_end_logits /= models_loaded

    # Post-process to get predictions
    example_to_indices = val_features_inf.groupby("example_id").indices
    predictions = {}

    # Disable verbose logging for pandas operations
    tqdm.pandas(disable=True)

    for _, row in tqdm(
        val_df.iterrows(), total=len(val_df), desc="Post-processing Validation"
    ):
        ex_id = row["id"]
        context_text = row["context"]

        if ex_id not in example_to_indices:
            predictions[ex_id] = ""
            continue

        feature_indices = example_to_indices[ex_id]
        best_score = -float("inf")
        best_answer = ""

        for idx in feature_indices:
            start_logit = agg_start_logits[idx]
            end_logit = agg_end_logits[idx]
            offsets = val_features_inf.iloc[idx]["offset_mapping"]
            token_type_ids = val_features_inf.iloc[idx]["token_type_ids"]

            if not isinstance(token_type_ids, np.ndarray):
                token_type_ids = np.array(token_type_ids)

            context_mask = token_type_ids == 1
            min_score = -1e9

            s_logits = np.where(context_mask, start_logit, min_score)
            e_logits = np.where(context_mask, end_logit, min_score)

            start_indexes = np.argsort(s_logits)[-Config.N_BEST_SIZE :][::-1]
            end_indexes = np.argsort(e_logits)[-Config.N_BEST_SIZE :][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index > end_index:
                        continue
                    if end_index - start_index + 1 > Config.MAX_ANSWER_LENGTH:
                        continue

                    score = start_logit[start_index] + end_logit[end_index]
                    if score > best_score:
                        best_score = score
                        try:
                            start_char = int(offsets[start_index][0])
                            end_char = int(offsets[end_index][1])
                            best_answer = context_text[start_char:end_char]
                        except:
                            continue

        predictions[ex_id] = best_answer

    # Compute Metric
    gt_answers = val_df.set_index("id")["answer_text"].to_dict()
    pred_list = [predictions[ex_id] for ex_id in val_df["id"]]
    gt_list = [gt_answers[ex_id] for ex_id in val_df["id"]]

    final_metric = compute_metric(gt_list, pred_list)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    jaccard_scores = [jaccard(gt, pred) for gt, pred in zip(gt_list, pred_list)]
    errors = [1.0 - s for s in jaccard_scores]

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "context_len": val_df["context"].apply(len),
            "question_len": val_df["question"].apply(len),
        }
    )

    corr_ctx = analysis_df["error"].corr(analysis_df["context_len"])
    corr_que = analysis_df["error"].corr(analysis_df["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
    print(f"Correlation (Error vs Question Length): {corr_que:.4f}")

    # 7. Submission
    THRESHOLD = 0.2522202380952381
    if final_metric > THRESHOLD:
        print("\nMetric exceeds threshold. Generating submission...")
        predictor.get_ensemble_predictions(folds=Config.NUM_FOLDS)
    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
