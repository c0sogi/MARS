import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import gc
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import AverageMeter, jaccard, normalize_text
from library.data import get_data_loaders, get_test_loader
from library.model import TweetModel


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    start_log_probs = F.log_softmax(start_logits, dim=1)
    end_log_probs = F.log_softmax(end_logits, dim=1)

    start_loss = loss_fct(start_log_probs, start_targets)
    end_loss = loss_fct(end_log_probs, end_targets)

    total_loss = (Config.LOSS_WEIGHT_START * start_loss) + (
        Config.LOSS_WEIGHT_END * end_loss
    )
    return total_loss


def get_optimizer_params(model):
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []
    lr = Config.LR_MAX
    decay = Config.LLRD_DECAY

    # 1. Head Parameters
    head_params = [p for n, p in named_parameters if "backbone" not in n]
    optimizer_parameters.append(
        {"params": head_params, "lr": lr, "weight_decay": Config.WEIGHT_DECAY}
    )

    # 2. Backbone Layers
    n_layers = model.model_config.num_hidden_layers
    for layer_i in range(n_layers - 1, -1, -1):
        layer_lr = lr * (decay ** (n_layers - layer_i))
        decay_params = [
            p
            for n, p in named_parameters
            if f"encoder.layer.{layer_i}." in n and not any(nd in n for nd in no_decay)
        ]
        no_decay_params = [
            p
            for n, p in named_parameters
            if f"encoder.layer.{layer_i}." in n and any(nd in n for nd in no_decay)
        ]

        if decay_params:
            optimizer_parameters.append(
                {
                    "params": decay_params,
                    "lr": layer_lr,
                    "weight_decay": Config.WEIGHT_DECAY,
                }
            )
        if no_decay_params:
            optimizer_parameters.append(
                {"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0}
            )

    # 3. Embeddings
    embed_lr = lr * (decay ** (n_layers + 1))
    decay_embed = [
        p
        for n, p in named_parameters
        if ("embeddings" in n or "rel_embeddings" in n)
        and not any(nd in n for nd in no_decay)
    ]
    no_decay_embed = [
        p
        for n, p in named_parameters
        if ("embeddings" in n or "rel_embeddings" in n)
        and any(nd in n for nd in no_decay)
    ]

    if decay_embed:
        optimizer_parameters.append(
            {"params": decay_embed, "lr": embed_lr, "weight_decay": Config.WEIGHT_DECAY}
        )
    if no_decay_embed:
        optimizer_parameters.append(
            {"params": no_decay_embed, "lr": embed_lr, "weight_decay": 0.0}
        )

    return optimizer_parameters


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        start_targets = d["start_targets"].to(device)
        end_targets = d["end_targets"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device, df_val):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    all_start_logits = []
    all_end_logits = []
    all_offsets = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            start_targets = d["start_targets"].to(device)
            end_targets = d["end_targets"].to(device)
            offsets = d["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
            losses.update(loss.item(), input_ids.size(0))

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())
            all_offsets.append(offsets)

    start_preds = np.concatenate(all_start_logits)
    end_preds = np.concatenate(all_end_logits)
    offsets_preds = np.concatenate(all_offsets)

    # Calculate Jaccard
    for i, row in df_val.iterrows():
        s_logits = start_preds[i]
        e_logits = end_preds[i]
        offsets = offsets_preds[i]

        # Joint Logit Decoding
        sum_logits = np.add.outer(s_logits, e_logits)
        mask = np.triu(np.ones_like(sum_logits))
        sum_logits[mask == 0] = -float("inf")

        best_idx = np.argmax(sum_logits)
        best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

        text = normalize_text(str(row["text"]))

        if best_start < len(offsets) and best_end < len(offsets):
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            pred_text = text[start_char:end_char]
        else:
            pred_text = text

        score = jaccard(pred_text, normalize_text(str(row["selected_text"])))
        jaccards.update(score)

    return losses.avg, jaccards.avg


def inference_fn_ensemble(loader, models, device, df):
    for model in models:
        model.eval()

    all_start_logits = []
    all_end_logits = []
    all_offsets = []

    with torch.no_grad():
        for d in loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            offsets = d["offsets"].cpu().numpy()

            # Accumulate logits from all models
            avg_start_logits = None
            avg_end_logits = None

            for model in models:
                start_logits, end_logits = model(input_ids, attention_mask)

                if avg_start_logits is None:
                    avg_start_logits = start_logits.cpu().numpy()
                    avg_end_logits = end_logits.cpu().numpy()
                else:
                    avg_start_logits += start_logits.cpu().numpy()
                    avg_end_logits += end_logits.cpu().numpy()

            # Average
            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            all_start_logits.append(avg_start_logits)
            all_end_logits.append(avg_end_logits)
            all_offsets.append(offsets)

    start_preds = np.concatenate(all_start_logits)
    end_preds = np.concatenate(all_end_logits)
    offsets_preds = np.concatenate(all_offsets)

    final_predictions = []

    for i, row in df.iterrows():
        text = normalize_text(str(row["text"]))
        sentiment = str(row["sentiment"])

        # Neutral Strategy: Return full text
        if sentiment == "neutral":
            final_predictions.append(f'"{text}"')
            continue

        s_logits = start_preds[i]
        e_logits = end_preds[i]
        offsets = offsets_preds[i]

        sum_logits = np.add.outer(s_logits, e_logits)
        mask = np.triu(np.ones_like(sum_logits))
        sum_logits[mask == 0] = -float("inf")

        best_idx = np.argmax(sum_logits)
        best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

        if best_start < len(offsets) and best_end < len(offsets):
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            pred_text = text[start_char:end_char]
        else:
            pred_text = text

        final_predictions.append(f'"{pred_text}"')

    return final_predictions


def run_experiment():
    print("Initializing Experiment...")
    device = Config.DEVICE

    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        train_loader, val_loader = get_data_loaders(
            fold=fold, load_cached_data=True, debug=Config.DEBUG
        )

        # Reconstruct validation dataframe for evaluation
        train_df = pd.read_csv(Config.TRAIN_META)
        if Config.DEBUG:
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )
        splits = list(skf.split(train_df, train_df["sentiment"]))
        _, val_idx = splits[fold]

        if Config.FILTER_NEUTRAL:
            is_not_neutral = train_df["sentiment"] != "neutral"
            val_subset_mask = is_not_neutral.iloc[val_idx].values
            val_idx = val_idx[val_subset_mask]

        val_df_fold = train_df.iloc[val_idx].reset_index(drop=True)

        model = TweetModel()
        model.to(device)

        optimizer_grouped_parameters = get_optimizer_params(model)
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=Config.LR_MAX, eps=Config.EPS
        )

        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        num_warmup_steps = int(num_train_steps * Config.NUM_WARMUP_STEPS_RATIO)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        best_jaccard = 0
        patience = 2
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
            val_loss, val_jaccard = eval_fn(val_loader, model, device, val_df_fold)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Jaccard: {val_jaccard}"
            )

            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                torch.save(model.state_dict(), Config.MODEL_PATH)
                print(f"Saved Best Model (Jaccard: {best_jaccard})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

        # Break after first fold as per strategy
        break

    print("\nGenerating Submission...")
    model = TweetModel()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)

    test_loader, test_df = get_test_loader(load_cached_data=True, debug=Config.DEBUG)
    predictions = inference_fn(test_loader, model, device, test_df)

    submission_df = pd.DataFrame(
        {"textID": test_df["textID"], "selected_text": predictions}
    )
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
