import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import AverageMeter, compute_metric, set_seed


class ToxicityModel(nn.Module):
    """
    Transformer-based model for multi-label toxicity classification.
    Uses a pre-trained backbone (e.g., RoBERTa) and a linear classification head.
    """

    def __init__(self, model_name, num_classes):
        super(ToxicityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, ids, mask, token_type_ids=None):
        # Handle inputs for different transformer architectures
        if token_type_ids is not None:
            out = self.bert(
                input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
            )
        else:
            out = self.bert(input_ids=ids, attention_mask=mask)

        # Use the CLS token embedding (first token of the last hidden state)
        # This is generally safer than pooler_output for models like RoBERTa
        cls_embeddings = out.last_hidden_state[:, 0, :]

        output = self.drop(cls_embeddings)
        output = self.fc(output)
        return output


class TransformerTrainer:
    """
    Manages the training, validation, and inference of the Transformer model.
    Implements Branch B of the hybrid solution.
    """

    def __init__(self, model_name, save_filename):
        self.device = Config.DEVICE
        self.model_name = model_name
        self.num_classes = len(Config.LABEL_COLS)
        self.learning_rate = Config.LEARNING_RATE
        self.epochs = Config.EPOCHS
        self.working_dir = Config.WORKING_DIR

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.working_dir, save_filename)

        # Initialize Model
        print(f"Initializing Transformer Model: {self.model_name}")
        self.model = ToxicityModel(self.model_name, self.num_classes)
        self.model.to(self.device)

        # Loss Function (Multi-label binary classification)
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_optimizer_and_scheduler(self, num_train_steps):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        return optimizer, scheduler

    def train_one_epoch(self, train_loader, optimizer, scheduler, scaler, epoch):
        self.model.train()
        losses = AverageMeter()

        print(f"Training Epoch {epoch + 1}/{self.epochs}...")

        for step, data in enumerate(train_loader):
            ids = data["ids"].to(self.device, dtype=torch.long)
            mask = data["mask"].to(self.device, dtype=torch.long)
            targets = data["targets"].to(self.device, dtype=torch.float)

            token_type_ids = None
            if "token_type_ids" in data:
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )

            optimizer.zero_grad()

            with autocast():
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.criterion(outputs, targets)

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.update(loss.item(), ids.size(0))

        print(f"Epoch {epoch + 1} Training Loss: {losses.avg}")
        return losses.avg

    def evaluate(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        final_targets = []
        final_outputs = []

        print("Running Validation...")

        with torch.no_grad():
            for data in val_loader:
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)
                targets = data["targets"].to(self.device, dtype=torch.float)

                token_type_ids = None
                if "token_type_ids" in data:
                    token_type_ids = data["token_type_ids"].to(
                        self.device, dtype=torch.long
                    )

                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), ids.size(0))

                # Move to CPU for metric calculation
                final_targets.extend(targets.cpu().detach().numpy().tolist())
                final_outputs.extend(
                    torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                )

        final_targets = np.array(final_targets)
        final_outputs = np.array(final_outputs)

        auc_score = compute_metric(final_targets, final_outputs)
        print(f"Validation Loss: {losses.avg}")
        print(f"Validation AUC: {auc_score}")

        return auc_score, losses.avg

    def train(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        num_train_steps = int(len(train_loader) * self.epochs)
        optimizer, scheduler = self._get_optimizer_and_scheduler(num_train_steps)
        scaler = GradScaler()

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self.train_one_epoch(train_loader, optimizer, scheduler, scaler, epoch)

            # Validate
            val_auc, val_loss = self.evaluate(val_loader)

            # Early Stopping and Checkpointing
            if val_auc > best_auc:
                print(f"AUC improved from {best_auc} to {val_auc}. Saving model...")
                best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"AUC did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Best Validation AUC: {best_auc}")
        return best_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Starting Inference...")

        # Load best model
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()
        final_outputs = []

        with torch.no_grad():
            for data in test_loader:
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)

                token_type_ids = None
                if "token_type_ids" in data:
                    token_type_ids = data["token_type_ids"].to(
                        self.device, dtype=torch.long
                    )

                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)

                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs).cpu().detach().numpy()
                final_outputs.extend(probs.tolist())

        print("Inference complete.")
        return np.array(final_outputs)
