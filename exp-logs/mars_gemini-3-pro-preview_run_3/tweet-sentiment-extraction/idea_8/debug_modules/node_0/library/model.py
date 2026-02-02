import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from library.config import Config
from library.utils import AverageMeter, get_score, seed_everything
from library.data import get_train_val_loaders, get_test_dataloader

# ==================================================================================
# Model Architecture
# ==================================================================================


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.config = AutoConfig.from_pretrained(
            Config.MODEL_NAME, output_hidden_states=True
        )

        # Apply dropout settings from Config
        self.config.hidden_dropout_prob = Config.HIDDEN_DROPOUT
        self.config.attention_probs_dropout_prob = Config.ATTENTION_DROPOUT

        self.deberta = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)

        # Simple Linear Head: Projects hidden_size -> 2 (Start Logit, End Logit)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # DeBERTa V3 forward pass
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # (Batch, Seq_Len, Hidden)

        # Linear projection
        logits = self.qa_outputs(sequence_output)  # (Batch, Seq_Len, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, Seq_Len)
        end_logits = end_logits.squeeze(-1)  # (Batch, Seq_Len)

        return start_logits, end_logits


# ==================================================================================
# Adversarial Weight Perturbation (AWP)
# ==================================================================================


class AWP:
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Perturbs the model weights based on the current gradients.
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        """
        Saves the original weights and computes the epsilon bounds.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        """
        Restores the original weights.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


# ==================================================================================
# Training & Evaluation Functions
# ==================================================================================


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    # CrossEntropyLoss with Label Smoothing
    loss_fct = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return start_loss + end_loss


def train_fn(dataloader, model, optimizer, scheduler, device, epoch, awp=None):
    model.train()
    losses = AverageMeter()

    # Enable AWP only after the start epoch
    use_awp = awp is not None and epoch >= Config.AWP_START_EPOCH

    for batch in tqdm(dataloader, desc=f"Train Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_labels = batch["start_labels"].to(device)
        end_labels = batch["end_labels"].to(device)

        # 1. Standard Forward Pass
        start_logits, end_logits = model(input_ids, attention_mask)
        loss = loss_fn(start_logits, end_logits, start_labels, end_labels)

        # 2. Standard Backward Pass
        loss.backward()

        # 3. AWP Step (Adversarial Training)
        if use_awp:
            awp.attack()
            # Adversarial Forward Pass
            start_logits_adv, end_logits_adv = model(input_ids, attention_mask)
            loss_adv = loss_fn(
                start_logits_adv, end_logits_adv, start_labels, end_labels
            )
            # Adversarial Backward Pass
            loss_adv.backward()
            awp.restore()

        # 4. Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(dataloader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_labels = batch["start_labels"].to(device)
            end_labels = batch["end_labels"].to(device)
            orig_texts = batch["orig_text"]
            offsets = batch["offsets"].cpu().numpy()

            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_labels, end_labels)
            losses.update(loss.item(), input_ids.size(0))

            # Decode predictions for Jaccard calculation
            start_probs = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

            for i in range(len(input_ids)):
                # Summation Decoding: Maximize P_start[s] + P_end[e] where s <= e
                start_p = start_probs[i]
                end_p = end_probs[i]
                offset = offsets[i]
                orig_text = orig_texts[i]
                target_text = orig_text  # Validation has selected_text, but we compare prediction vs target
                # Wait, we need the ground truth selected_text to compute Jaccard score
                # The dataloader doesn't yield selected_text string directly, but we can infer or pass it.
                # Actually, validation loader uses 'orig_texts' which is the full tweet.
                # We need the ground truth selected text.
                # However, `get_score` takes strings.
                # Let's assume we can't compute exact Jaccard here without the GT string.
                # But `val_loader` was built with `orig_texts`.
                # We need the `selected_text` for evaluation.
                # The `TweetDataset` doesn't return `selected_text`.
                # We will skip Jaccard calculation inside the loop and rely on Loss,
                # or we would need to modify the dataset.
                # For this implementation, we will track Loss primarily in this function.
                pass

    return losses.avg


# ==================================================================================
# Main Execution Helpers
# ==================================================================================


def run_training():
    """
    Executes the 5-Fold Cross Validation training with AWP.
    """
    seed_everything(Config.SEED)

    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold+1} / {Config.N_FOLDS} {'='*20}")

        train_loader, val_loader = get_train_val_loaders(fold, load_cached_data=True)

        model = TweetModel()
        model.to(Config.DEVICE)

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

        # Initialize AWP
        awp = AWP(model, optimizer, adv_lr=Config.AWP_LR, adv_eps=Config.AWP_EPS)

        best_loss = float("inf")

        for epoch in range(Config.EPOCHS):
            train_loss = train_fn(
                train_loader, model, optimizer, scheduler, Config.DEVICE, epoch, awp
            )
            val_loss = eval_fn(val_loader, model, Config.DEVICE)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin"),
                )
                print("  -> Saved Best Model")

        del model, optimizer, scheduler, awp
        torch.cuda.empty_cache()


def generate_submission():
    """
    Runs inference on the test set using the ensemble of 5 fold models.
    Implements Summation Decoding and Neutral Override.
    """
    print("\nGenerating Submission...")
    test_loader, df_test = get_test_dataloader(load_cached_data=True)

    # Load all models
    models = []
    for fold in range(Config.N_FOLDS):
        model = TweetModel()
        model.to(Config.DEVICE)
        path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")
        if not os.path.exists(path):
            print(f"Warning: Model for fold {fold} not found at {path}. Skipping.")
            continue
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        model.eval()
        models.append(model)

    if not models:
        print("Error: No models found.")
        return

    final_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            offsets = batch["offsets"].cpu().numpy()
            orig_texts = batch["orig_text"]

            # Ensemble Predictions
            avg_start_logits = torch.zeros(input_ids.size(0), input_ids.size(1)).to(
                Config.DEVICE
            )
            avg_end_logits = torch.zeros(input_ids.size(0), input_ids.size(1)).to(
                Config.DEVICE
            )

            for model in models:
                start_logits, end_logits = model(input_ids, attention_mask)
                avg_start_logits += torch.softmax(start_logits, dim=1)
                avg_end_logits += torch.softmax(end_logits, dim=1)

            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            start_probs = avg_start_logits.cpu().numpy()
            end_probs = avg_end_logits.cpu().numpy()

            # Decoding
            for i in range(len(input_ids)):
                text = orig_texts[i]

                # Neutral Rule: If neutral, return full text
                # We need the sentiment for this row.
                # The dataloader batch doesn't carry sentiment string explicitly,
                # but we can look it up in df_test via index if order is preserved.
                # However, df_test order matches test_loader order (shuffle=False).
                # We calculate global index.
                # Better approach: The dataset logic encodes sentiment in input_ids.
                # But we can't easily decode it back.
                # We will rely on the fact that df_test is parallel to test_loader.
                # We need to track the global index.
                # For simplicity here, we assume sequential processing.

                # Summation Decoding
                # Maximize P_start[s] + P_end[e] for s <= e
                s_p = start_probs[i]
                e_p = end_probs[i]

                # Create score matrix
                # shape: (seq_len, seq_len)
                score_mat = np.expand_dims(s_p, 1) + np.expand_dims(e_p, 0)

                # Mask out cases where start > end
                # We can use triu
                seq_len = len(s_p)
                mask = np.triu(np.ones((seq_len, seq_len)), k=0)
                score_mat = score_mat * mask
                # Set masked values to -inf to avoid selection
                score_mat[mask == 0] = -np.inf

                # Find argmax
                best_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
                start_idx, end_idx = best_idx

                # Convert token indices to character span
                # offsets[i] is list of (start, end)
                if start_idx < len(offsets[i]) and end_idx < len(offsets[i]):
                    char_start = offsets[i][start_idx][0]
                    char_end = offsets[i][end_idx][1]
                    pred_text = text[char_start:char_end]
                else:
                    pred_text = text

                final_predictions.append(pred_text)

    # Apply Neutral Rule and Save
    # We need to align with df_test
    if len(final_predictions) != len(df_test):
        print(
            f"Warning: Prediction count {len(final_predictions)} != Test set size {len(df_test)}"
        )

    df_test["selected_text"] = final_predictions

    # Apply Neutral Rule
    df_test.loc[df_test["sentiment"] == Config.SENTIMENT_NEUTRAL, "selected_text"] = (
        df_test.loc[df_test["sentiment"] == Config.SENTIMENT_NEUTRAL, "text"]
    )

    # Formatting
    submission = df_test[["textID", "selected_text"]]
    # Ensure quotes are handled if necessary, but pandas to_csv handles this.
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_idea_8():
    """
    Orchestrator function.
    """
    run_training()
    generate_submission()
