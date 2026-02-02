import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.data import get_data, TweetDataset, TweetTestDataset
from library.model import TweetModel
from library.engine import eval_fn
from library.utils import seed_everything, AverageMeter, jaccard

# Suppress warnings
warnings.filterwarnings("ignore")


def run():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Speed/Efficiency on A100
    # Increase to 3 Epochs with LLRD to allow deeper layers to adapt slowly
    Config.EPOCHS = 3
    Config.TRAIN_BATCH_SIZE = 32
    Config.VALID_BATCH_SIZE = 64
    # Slightly higher base LR for the head/top layers to compensate for decay
    Config.LEARNING_RATE = 2.5e-5

    device = Config.DEVICE

    # Create submission directory
    if not os.path.exists("./submission"):
        os.makedirs("./submission")

    # 2. Data Loading
    train_data, test_data = get_data(load_cached_data=True)

    # 3. Cross-Validation Loop
    val_results = []

    # Layer-wise Learning Rate Decay (LLRD) Helper
    def get_optimizer_params(model, base_lr, weight_decay=0.01, decay_rate=0.9):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = []

        # 1. Head Parameters (Classifier)
        head_params = [(n, p) for n, p in param_optimizer if "backbone" not in n]
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": base_lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": base_lr,
            }
        )

        # 2. Backbone Parameters
        num_layers = 24

        # Embeddings (Lowest LR)
        embed_params = [(n, p) for n, p in param_optimizer if "embeddings" in n]
        embed_lr = base_lr * (decay_rate ** (num_layers + 1))
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in embed_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": embed_lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in embed_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": embed_lr,
            }
        )

        # Encoder Layers
        for i in range(num_layers):
            # layer.0 is bottom, layer.23 is top
            layer_params = [
                (n, p) for n, p in param_optimizer if f"encoder.layer.{i}." in n
            ]
            # Decay relative to top.
            layer_lr = base_lr * (decay_rate ** (num_layers - i))

            optimizer_parameters.append(
                {
                    "params": [
                        p
                        for n, p in layer_params
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": layer_lr,
                }
            )
            optimizer_parameters.append(
                {
                    "params": [
                        p for n, p in layer_params if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": layer_lr,
                }
            )

        return optimizer_parameters

    for fold in range(Config.N_FOLDS):
        # Split indices
        train_idx = np.where(train_data["folds"] != fold)[0]
        val_idx = np.where(train_data["folds"] == fold)[0]

        # Create Datasets
        train_dataset = TweetDataset(train_data, indices=train_idx)
        val_dataset = TweetDataset(train_data, indices=val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Optimizer
        model = TweetModel()
        model.to(device)

        optimizer_grouped_parameters = get_optimizer_params(
            model,
            base_lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            decay_rate=0.9,
        )

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_train_steps = int(
            len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Mixed Precision Scaler
        scaler = torch.cuda.amp.GradScaler()

        # Training Loop
        best_jaccard = 0
        model_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS):
            model.train()
            losses = AverageMeter()

            for data in train_loader:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                start_targets = data["start_targets"].to(device)
                end_targets = data["end_targets"].to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    start_logits, end_logits = model(input_ids, attention_mask)
                    loss_fct = nn.CrossEntropyLoss(
                        label_smoothing=Config.LABEL_SMOOTHING
                    )
                    loss = loss_fct(start_logits, start_targets) + loss_fct(
                        end_logits, end_targets
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                losses.update(loss.item(), input_ids.size(0))

            # Evaluation
            val_loss, val_jaccard = eval_fn(val_loader, model, device)
            print(
                f"Fold {fold} | Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Val Jaccard: {val_jaccard:.4f}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), model_path)

        if not os.path.exists(model_path):
            print(
                f"Fold {fold} - No model saved (Best Jaccard: {best_jaccard}). Skipping OOF."
            )
            continue

        # Load best model for OOF generation
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Generate OOF predictions for this fold
        with torch.no_grad():
            for data in val_loader:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                offsets = data["offsets"].cpu().numpy()
                orig_texts = data["orig_text"]
                sentiments = data["sentiment"]

                start_targets = data["start_targets"].cpu().numpy()
                end_targets = data["end_targets"].cpu().numpy()

                start_logits, end_logits = model(input_ids, attention_mask)
                start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                for i in range(len(input_ids)):
                    text = orig_texts[i]
                    sentiment = sentiments[i]
                    offset = offsets[i]

                    # GT Reconstruction
                    s_idx_gt = start_targets[i]
                    e_idx_gt = end_targets[i]
                    if (
                        s_idx_gt < len(offset)
                        and e_idx_gt < len(offset)
                        and offset[s_idx_gt][0] <= offset[e_idx_gt][1]
                    ):
                        gt_text = text[offset[s_idx_gt][0] : offset[e_idx_gt][1]]
                    else:
                        gt_text = text

                    # Prediction Decoding
                    if sentiment == "neutral":
                        pred_text = text
                    else:
                        idx_start = np.argmax(start_probs[i])
                        idx_end = np.argmax(end_probs[i])
                        if idx_start > idx_end:
                            idx_end = idx_start
                        pred_text = text[offset[idx_start][0] : offset[idx_end][1]]

                    score = jaccard(pred_text, gt_text)

                    val_results.append(
                        {
                            "text": text,
                            "sentiment": sentiment,
                            "gt_text": gt_text,
                            "pred_text": pred_text,
                            "jaccard": score,
                            "text_len": len(text.split()),
                        }
                    )

        # Cleanup
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    # 4. Final Validation Metric
    df_results = pd.DataFrame(val_results)
    final_metric = df_results["jaccard"].mean()
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_results["error"] = 1.0 - df_results["jaccard"]
    corr_len = np.corrcoef(df_results["text_len"], df_results["error"])[0, 1]
    print(f"Correlation (Text Length vs Error): {corr_len:.4f}")

    print("Mean Jaccard by Sentiment:")
    print(df_results.groupby("sentiment")["jaccard"].mean())

    # 6. Inference & Submission
    THRESHOLD = 0.7205

    if final_metric > THRESHOLD:
        test_dataset = TweetTestDataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        num_test = len(test_dataset)
        seq_len = Config.MAX_LEN

        # Accumulators for Ensemble
        avg_start_logits = np.zeros((num_test, seq_len), dtype=np.float32)
        avg_end_logits = np.zeros((num_test, seq_len), dtype=np.float32)

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth")
            if not os.path.exists(model_path):
                print(f"Skipping Fold {fold} inference (model not found)")
                continue

            model = TweetModel()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            batch_start_idx = 0

            with torch.no_grad():
                for data in test_loader:
                    input_ids = data["input_ids"].to(device)
                    attention_mask = data["attention_mask"].to(device)

                    with torch.cuda.amp.autocast():
                        start_logits, end_logits = model(input_ids, attention_mask)

                    bs = input_ids.size(0)
                    avg_start_logits[batch_start_idx : batch_start_idx + bs] += (
                        start_logits.float().cpu().numpy()
                    )
                    avg_end_logits[batch_start_idx : batch_start_idx + bs] += (
                        end_logits.float().cpu().numpy()
                    )

                    batch_start_idx += bs

            del model
            torch.cuda.empty_cache()

        # Decode
        predictions = []
        for i in range(num_test):
            item = test_dataset[i]
            text = item["orig_text"]
            sentiment = item["sentiment"]
            offset = item["offsets"].numpy()
            t_id = item["text_id"]

            if sentiment == "neutral":
                pred_text = text
            else:
                start_l = avg_start_logits[i]
                end_l = avg_end_logits[i]

                idx_start = np.argmax(start_l)
                idx_end = np.argmax(end_l)

                if idx_start > idx_end:
                    idx_end = idx_start

                pred_text = text[offset[idx_start][0] : offset[idx_end][1]]

            predictions.append({"textID": t_id, "selected_text": pred_text})

        sub_df = pd.DataFrame(predictions)
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    run()
