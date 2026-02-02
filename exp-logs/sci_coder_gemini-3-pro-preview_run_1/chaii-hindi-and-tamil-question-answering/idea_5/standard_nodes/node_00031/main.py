import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import collections
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
from library.configuration import Config
from library.utilities import set_seed, jaccard
from library.data_processing import (
    prepare_train_features,
    prepare_test_features,
    QADataset,
)
from library.modeling import MultiTaskQAModel


def run_training(config, train_df):
    """
    Trains the model on the provided training DataFrame.
    """
    print("Processing training features...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_features = prepare_train_features(train_df, tokenizer, config)
    train_dataset = QADataset(train_features, mode="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = MultiTaskQAModel(config)
    model.to(config.device)

    # Differential Learning Rates: Higher LR for heads, Base LR for backbone
    # This helps the randomly initialized heads converge faster while preserving backbone features.
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.backbone.named_parameters()
                if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
            "lr": config.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate * 5,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    num_train_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    span_loss_fct = nn.CrossEntropyLoss()
    relevance_loss_fct = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    print(f"Starting training for {config.epochs} epochs...")
    model.train()

    for epoch in range(config.epochs):
        total_loss = 0.0
        # Disable tqdm for silent execution requirement, or keep minimal
        for batch in train_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            start_positions = batch["start_positions"].to(config.device)
            end_positions = batch["end_positions"].to(config.device)
            relevance_labels = batch["relevance_labels"].to(config.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                start_logits, end_logits, relevance_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                start_loss = span_loss_fct(start_logits, start_positions)
                end_loss = span_loss_fct(end_logits, end_positions)
                span_loss = (start_loss + end_loss) / 2
                rel_loss = relevance_loss_fct(relevance_logits, relevance_labels)

                loss = span_loss + (config.relevance_loss_weight * rel_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    return model


def run_inference(config, model, df, mode="val"):
    """
    Runs inference on a DataFrame (val or test).
    Returns a dictionary of predictions {id: prediction_string}.
    """
    print(f"Running inference on {mode} set...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Prepare features (treat val as test to get offsets)
    features = prepare_test_features(df, tokenizer, config)
    dataset = QADataset(features, mode="test")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model.eval()

    # Map ID to Context for extraction
    id_to_context = dict(zip(df["id"], df["context"]))
    all_predictions = collections.defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            offset_mapping = batch["offset_mapping"].numpy()
            example_ids = batch["example_id"]

            start_logits, end_logits, relevance_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            relevance_logits = relevance_logits.cpu().numpy()
            input_ids_cpu = input_ids.cpu().numpy()

            for i, example_id in enumerate(example_ids):
                # Identify Context Tokens (XLM-R)
                sep_indices = np.where(input_ids_cpu[i] == tokenizer.sep_token_id)[0]
                # <s> Q </s> </s> C </s>
                if len(sep_indices) >= 2:
                    context_start_idx = sep_indices[1] + 1
                    if len(sep_indices) >= 3:
                        context_end_idx = sep_indices[2] - 1
                    else:
                        context_end_idx = len(input_ids_cpu[i]) - 2
                else:
                    context_start_idx = 0
                    context_end_idx = len(input_ids_cpu[i]) - 1

                s_logits = start_logits[i]
                e_logits = end_logits[i]
                rel_score = relevance_logits[i]
                offsets = offset_mapping[i]

                best_score = -float("inf")
                best_text = ""

                start_indexes = np.argsort(s_logits)[-config.n_best_size :]
                end_indexes = np.argsort(e_logits)[-config.n_best_size :]

                for start_index in start_indexes:
                    if start_index < context_start_idx or start_index > context_end_idx:
                        continue
                    for end_index in end_indexes:
                        if end_index < context_start_idx or end_index > context_end_idx:
                            continue
                        if end_index < start_index:
                            continue
                        if end_index - start_index + 1 > config.max_answer_length:
                            continue

                        score = s_logits[start_index] + e_logits[end_index] + rel_score
                        if score > best_score:
                            best_score = score
                            start_char = offsets[start_index][0]
                            end_char = offsets[end_index][1]
                            context = id_to_context.get(example_id, "")
                            best_text = context[start_char:end_char]

                if best_text:
                    all_predictions[example_id].append((best_score, best_text))

    # Aggregate predictions
    final_preds = {}
    for eid in df["id"].unique():
        preds = all_predictions.get(eid, [])
        if not preds:
            final_preds[eid] = ""
        else:
            preds.sort(key=lambda x: x[0], reverse=True)
            final_preds[eid] = preds[0][1]

    return final_preds


def main():
    config = Config()
    set_seed(config.seed)

    # 1. Load Data
    print("Loading metadata...")
    train_df = pd.read_csv(config.train_meta_path)
    val_df = pd.read_csv(config.val_meta_path)

    # 2. Train
    model = run_training(config, train_df)

    # 3. Validation
    val_preds_map = run_inference(config, model, val_df, mode="val")

    # Compute Metric
    scores = []
    val_df["prediction"] = val_df["id"].map(val_preds_map)
    val_df["jaccard"] = val_df.apply(
        lambda row: jaccard(row["answer_text"], row["prediction"]), axis=1
    )

    final_metric = val_df["jaccard"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n==== Failure Analysis ====")
    val_df["error"] = 1.0 - val_df["jaccard"]
    val_df["context_len"] = val_df["context"].apply(len)
    val_df["question_len"] = val_df["question"].apply(len)

    corr_ctx = val_df["error"].corr(val_df["context_len"])
    corr_que = val_df["error"].corr(val_df["question_len"])

    print(f"Correlation (Error vs Context Length): {corr_ctx:.4f}")
    print(f"Correlation (Error vs Question Length): {corr_que:.4f}")

    # 5. Submission
    THRESHOLD = 0.5805833333333333

    if final_metric > THRESHOLD:
        print("\nMetric passed threshold. Generating submission...")
        test_df = pd.read_csv(config.test_meta_path)
        test_preds_map = run_inference(config, model, test_df, mode="test")

        submission_data = []
        for eid in test_df["id"].unique():
            # Ensure quoted format
            pred_text = test_preds_map.get(eid, "")
            formatted_pred = f'"{pred_text}"'
            submission_data.append({"id": eid, "PredictionString": formatted_pred})

        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(
            config.submission_path, index=False, quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        print(f"Submission saved to {config.submission_path}")
    else:
        print("\nMetric failed threshold. Skipping submission.")


if __name__ == "__main__":
    main()
