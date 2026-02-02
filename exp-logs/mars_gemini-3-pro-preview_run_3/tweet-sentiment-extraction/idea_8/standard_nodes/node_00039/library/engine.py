import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, AverageMeter, jaccard
from library.data import get_train_val_loaders, get_test_dataloader
from library.model import TweetModel, loss_fn
from library.awp import AWP


def train_fn(dataloader, model, optimizer, scheduler, device, epoch, awp=None):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    # Enable AWP only after the specific start epoch
    use_awp = (awp is not None) and (epoch >= Config.AWP_START_EPOCH)

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}/{Config.EPOCHS}", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_labels = batch["start_labels"].to(device)
        end_labels = batch["end_labels"].to(device)

        # 1. Standard Forward Pass
        start_logits, end_logits = model(input_ids, attention_mask)
        loss = loss_fn(start_logits, end_logits, start_labels, end_labels)

        # 2. Standard Backward Pass
        loss.backward()

        # 3. Adversarial Training (AWP)
        if use_awp:
            awp.attack()
            # Adversarial Forward Pass
            adv_start_logits, adv_end_logits = model(input_ids, attention_mask)
            adv_loss = loss_fn(
                adv_start_logits, adv_end_logits, start_labels, end_labels
            )

            # Adversarial Backward Pass
            adv_loss.backward()
            awp.restore()

        # 4. Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.update(loss.item(), input_ids.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

    return losses.avg


def eval_fn(dataloader, model, device, val_df):
    """
    Executes validation loop with Jaccard calculation.
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    # We iterate sequentially. Since val_loader is shuffle=False,
    # we can index into val_df sequentially.
    val_texts = val_df["text"].values
    val_selected = val_df["selected_text"].values
    val_sentiments = val_df["sentiment"].values

    batch_idx_start = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)
            offsets = batch["offsets"].cpu().numpy()

            batch_size = input_ids.size(0)

            # Forward
            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_labels, end_labels)
            losses.update(loss.item(), batch_size)

            # Decode
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(batch_size):
                # Retrieve ground truth info
                global_idx = batch_idx_start + i
                text = str(val_texts[global_idx])
                target_text = str(val_selected[global_idx])
                sentiment = str(val_sentiments[global_idx])

                # Summation Decoding: Maximize P_start + P_end
                s_p = start_probs[i]
                e_p = end_probs[i]

                # Create score matrix (seq_len, seq_len)
                # score[s, e] = s_p[s] + e_p[e]
                score_mat = np.expand_dims(s_p, 1) + np.expand_dims(e_p, 0)

                # Mask invalid spans (start > end)
                seq_len = len(s_p)
                mask = np.triu(np.ones((seq_len, seq_len)))
                score_mat = score_mat * mask + (1 - mask) * -1e10

                # Get best span
                best_idx = np.argmax(score_mat)
                start_idx, end_idx = np.unravel_index(best_idx, score_mat.shape)

                # Extract text using offsets
                current_offsets = offsets[i]
                if start_idx < len(current_offsets) and end_idx < len(current_offsets):
                    char_start = current_offsets[start_idx][0]
                    char_end = current_offsets[end_idx][1]
                    pred_text = text[char_start:char_end]
                else:
                    pred_text = text

                # Neutral Rule (though training data usually excludes neutral, validation might have it if not filtered?
                # The data loader filters neutrals for training.
                # We should check if validation set in loader has neutrals.
                # get_train_val_loaders filters neutrals from the dataframe before splitting.
                # So val_df here contains NO neutrals.
                pass

                score = jaccard(target_text, pred_text)
                jaccards.update(score, 1)

            batch_idx_start += batch_size

    return losses.avg, jaccards.avg


def run_training():
    """
    Main training loop for 5-Fold CV.
    """
    seed_everything(Config.SEED)

    # Load full training metadata to reconstruct validation sets for Jaccard calc
    full_df = pd.read_csv(Config.TRAIN_META_PATH)
    # Apply same filtering as in data.py
    full_df = full_df[full_df["sentiment"] != Config.SENTIMENT_NEUTRAL].reset_index(
        drop=True
    )

    # Alignment filtering
    valid_mask = full_df.apply(
        lambda x: str(x["selected_text"]) in str(x["text"]), axis=1
    )
    full_df = full_df[valid_mask].reset_index(drop=True)

    if Config().DEBUG:
        full_df = full_df.head(Config().DEBUG_SAMPLE_SIZE)

    # Ensure n_splits >= 2 even if we only want to run 1 fold (Config.N_FOLDS=1)
    n_splits = Config.N_FOLDS if Config.N_FOLDS > 1 else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
    splits = list(skf.split(full_df, full_df["sentiment"]))

    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold+1} / {Config.N_FOLDS} {'='*20}")

        # Get Loaders
        train_loader, val_loader = get_train_val_loaders(fold, load_cached_data=True)

        # Get Validation DataFrame for this fold
        _, val_idx = splits[fold]
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Init Model
        model = TweetModel()
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        # Init AWP
        awp = AWP(model, optimizer, adv_lr=Config.AWP_LR, adv_eps=Config.AWP_EPS)

        best_loss = float("inf")
        best_jaccard = 0.0

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(
                train_loader, model, optimizer, scheduler, Config.DEVICE, epoch, awp
            )
            val_loss, val_jaccard = eval_fn(val_loader, model, Config.DEVICE, val_df)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Jaccard: {val_jaccard:.5f}"
            )

            # Save Best Model (Monitoring Loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_jaccard = val_jaccard
                save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved Best Model (Loss: {best_loss:.5f})")

        print(f"Fold {fold+1} Best Jaccard: {best_jaccard:.5f}")

        # Cleanup
        del model, optimizer, scheduler, awp, train_loader, val_loader
        torch.cuda.empty_cache()


def generate_submission():
    """
    Inference on Test Set.
    """
    print("\nGenerating Submission...")
    test_loader, df_test = get_test_dataloader(load_cached_data=True)

    # Load Models
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")
        if os.path.exists(model_path):
            model = TweetModel()
            model.to(Config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model for fold {fold} not found.")

    if not models:
        print("Error: No models loaded.")
        return

    final_predictions = []

    # Inference Loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            offsets = batch["offsets"].cpu().numpy()
            orig_texts = batch["orig_text"]

            # Ensemble Averaging
            avg_start_logits = None
            avg_end_logits = None

            for model in models:
                start_logits, end_logits = model(input_ids, attention_mask)
                start_probs = torch.softmax(start_logits, dim=1)
                end_probs = torch.softmax(end_logits, dim=1)

                if avg_start_logits is None:
                    avg_start_logits = start_probs
                    avg_end_logits = end_probs
                else:
                    avg_start_logits += start_probs
                    avg_end_logits += end_probs

            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            avg_start_logits = avg_start_logits.cpu().numpy()
            avg_end_logits = avg_end_logits.cpu().numpy()

            # Decoding
            for i in range(len(input_ids)):
                text = orig_texts[i]
                s_p = avg_start_logits[i]
                e_p = avg_end_logits[i]

                # Summation Decoding
                score_mat = np.expand_dims(s_p, 1) + np.expand_dims(e_p, 0)
                seq_len = len(s_p)
                mask = np.triu(np.ones((seq_len, seq_len)))
                score_mat = score_mat * mask + (1 - mask) * -1e10

                best_idx = np.argmax(score_mat)
                start_idx, end_idx = np.unravel_index(best_idx, score_mat.shape)

                current_offsets = offsets[i]
                if start_idx < len(current_offsets) and end_idx < len(current_offsets):
                    char_start = current_offsets[start_idx][0]
                    char_end = current_offsets[end_idx][1]
                    pred_text = text[char_start:char_end]
                else:
                    pred_text = text

                final_predictions.append(pred_text)

    # Assign to dataframe
    df_test["selected_text"] = final_predictions

    # Apply Neutral Rule
    # If sentiment is neutral, selected_text = text
    df_test.loc[df_test["sentiment"] == Config.SENTIMENT_NEUTRAL, "selected_text"] = (
        df_test["text"]
    )

    # Save
    submission = df_test[["textID", "selected_text"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_idea_8():
    run_training()
    generate_submission()
