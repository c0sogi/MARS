import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from library.config import Config, seed_everything
from library.utils import compute_qwk, optimize_thresholds, apply_thresholds
from library.data_loader import get_dataloaders


class EssayRegressor(nn.Module):
    """
    Regression model based on DeBERTa-v3-base.
    Uses Mean Pooling on the last hidden state followed by a Linear layer.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # Enable gradient checkpointing to save memory
        self.backbone.gradient_checkpointing_enable()
        # Cite debug_lesson_1: Explicitly enable input gradients for checkpointing in custom loops
        self.backbone.enable_input_require_grads()

        # DeBERTa-v3-base hidden size is 768
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # outputs.last_hidden_state: (Batch, SeqLen, Hidden)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling with Attention Mask
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # (Batch, Hidden)

        # Regression Head
        logits = self.fc(mean_embeddings)  # (Batch, 1)
        return logits


def train_model(train_loader, val_loader):
    """
    Trains the EssayRegressor model.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.

    Returns:
        str: Path to the saved best model checkpoint.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Initializing model: {Config.MODEL_NAME} on {device}")

    model = EssayRegressor(Config.MODEL_NAME, pretrained=True).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")

    # Tracking
    best_qwk = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience = 2
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask).squeeze(-1)
                loss = criterion(outputs, labels)

                # Gradient Accumulation
                if Config.GRAD_ACCUMULATION_STEPS > 1:
                    loss = loss / Config.GRAD_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * Config.GRAD_ACCUMULATION_STEPS

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, attention_mask).squeeze(-1)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

                val_preds.extend(outputs.float().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Compute QWK for monitoring (using standard rounding)
        val_preds_np = np.array(val_preds)
        val_labels_np = np.array(val_labels)
        val_preds_rounded = np.clip(np.round(val_preds_np), 1, 6).astype(int)
        val_labels_int = val_labels_np.astype(int)

        epoch_qwk = compute_qwk(val_labels_int, val_preds_rounded)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val QWK (Rounded): {epoch_qwk:.6f}"
        )

        # Save Best Model
        if epoch_qwk > best_qwk:
            best_qwk = epoch_qwk
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! QWK: {best_qwk:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_model_path


def predict_and_submit(model_path, val_loader, test_loader):
    """
    Loads the best model, optimizes thresholds on validation data,
    predicts on test data, and generates the submission file.
    """
    device = Config.DEVICE
    print(f"Loading best model from {model_path} for inference...")

    # Initialize model structure without downloading weights, then load state dict
    model = EssayRegressor(Config.MODEL_NAME, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. Validation Inference for Threshold Optimization
    print("Generating validation predictions to optimize thresholds...")
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask).squeeze(-1)
            val_preds.extend(outputs.float().cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds_np = np.array(val_preds)
    val_labels_int = np.array(val_labels).astype(int)

    if Config.OPTIMIZE_THRESHOLDS:
        print("Optimizing thresholds using Nelder-Mead...")
        best_thresholds = optimize_thresholds(val_labels_int, val_preds_np)
        print(f"Optimized Thresholds: {best_thresholds}")
    else:
        print("Using standard rounding thresholds.")
        best_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    # Verify QWK with optimized thresholds
    val_preds_opt = apply_thresholds(val_preds_np, best_thresholds)
    opt_qwk = compute_qwk(val_labels_int, val_preds_opt)
    print(f"Validation QWK with optimized thresholds: {opt_qwk:.6f}")

    # 2. Test Inference
    print("Generating test predictions...")
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask).squeeze(-1)
            test_preds.extend(outputs.float().cpu().numpy())

    test_preds_np = np.array(test_preds)

    # Apply thresholds
    final_scores = apply_thresholds(test_preds_np, best_thresholds)

    # 3. Create Submission
    # We read the test metadata to ensure ID alignment
    df_test = pd.read_csv(Config.TEST_PATH)

    if len(df_test) != len(final_scores):
        print(
            f"WARNING: Mismatch in test set length. DF: {len(df_test)}, Preds: {len(final_scores)}"
        )

    submission = pd.DataFrame({"essay_id": df_test["essay_id"], "score": final_scores})

    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_pipeline():
    """
    Main entry point to run the full training and submission pipeline.
    """
    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Train
    best_model_path = train_model(train_loader, val_loader)

    # Predict and Submit
    predict_and_submit(best_model_path, val_loader, test_loader)
