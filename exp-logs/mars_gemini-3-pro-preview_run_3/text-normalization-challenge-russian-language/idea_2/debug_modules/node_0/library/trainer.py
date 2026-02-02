import torch
import torch.nn as nn
import time
import sys
from library.config import Config
from library.utils import MetricTracker, save_checkpoint


class Trainer:
    """
    Trainer class for the Text Normalization Transformer model.
    Handles training, validation, and checkpointing.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device=Config.DEVICE,
    ):
        """
        Args:
            model: The TransformerNumNorm model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: PyTorch optimizer.
            scheduler: PyTorch learning rate scheduler.
            device: Computing device (CPU or GPU).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Loss Functions
        # Text Generation: Ignore padding tokens
        self.criterion_text = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
        # Auxiliary Classification: Standard Cross Entropy
        self.criterion_class = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()

        # Trackers
        loss_tracker = MetricTracker()
        text_loss_tracker = MetricTracker()
        class_loss_tracker = MetricTracker()
        text_acc_tracker = MetricTracker()
        class_acc_tracker = MetricTracker()

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            src = batch["src"].to(self.device)
            tgt_in = batch["tgt_in"].to(self.device)
            tgt_out = batch["tgt_out"].to(self.device)
            class_id = batch["class_id"].to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            # text_logits: (batch, seq_len, vocab)
            # class_logits: (batch, num_classes)
            text_logits, class_logits = self.model(src, tgt_in)

            # Reshape for Loss
            # text_logits -> (batch * seq_len, vocab)
            # tgt_out -> (batch * seq_len)
            text_logits_flat = text_logits.view(-1, text_logits.size(-1))
            tgt_out_flat = tgt_out.view(-1)

            # Calculate Losses
            loss_text = self.criterion_text(text_logits_flat, tgt_out_flat)
            loss_class = self.criterion_class(class_logits, class_id)

            # Composite Loss
            total_loss = loss_text + (Config.LAMBDA_CLASS_LOSS * loss_class)

            # Backward Pass
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            self.optimizer.step()

            # Step scheduler if it's step-based (e.g., OneCycleLR or Transformer schedule)
            # Assuming epoch-based scheduler for this implementation, but if it was step-based:
            # if self.scheduler: self.scheduler.step()

            # --- Metrics ---
            batch_size = src.size(0)

            # Text Accuracy (Token-level, ignoring pad)
            preds_text = torch.argmax(text_logits_flat, dim=1)
            mask = tgt_out_flat != Config.PAD_IDX
            correct_text = (preds_text[mask] == tgt_out_flat[mask]).sum().item()
            total_text = mask.sum().item()
            text_acc = correct_text / total_text if total_text > 0 else 0.0

            # Class Accuracy
            preds_class = torch.argmax(class_logits, dim=1)
            correct_class = (preds_class == class_id).sum().item()
            class_acc = correct_class / batch_size

            # Update Trackers
            loss_tracker.update(total_loss.item(), batch_size)
            text_loss_tracker.update(loss_text.item(), batch_size)
            class_loss_tracker.update(loss_class.item(), batch_size)
            text_acc_tracker.update(text_acc, batch_size)
            class_acc_tracker.update(class_acc, batch_size)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} [Train] Loss: {loss_tracker.avg} | Text Loss: {text_loss_tracker.avg} | Class Loss: {class_loss_tracker.avg} | Text Acc: {text_acc_tracker.avg} | Class Acc: {class_acc_tracker.avg} | Time: {elapsed:.2f}s"
        )

        return loss_tracker.avg

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        """
        self.model.eval()

        loss_tracker = MetricTracker()
        text_loss_tracker = MetricTracker()
        class_loss_tracker = MetricTracker()
        text_acc_tracker = MetricTracker()
        class_acc_tracker = MetricTracker()

        start_time = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["src"].to(self.device)
                tgt_in = batch["tgt_in"].to(self.device)
                tgt_out = batch["tgt_out"].to(self.device)
                class_id = batch["class_id"].to(self.device)

                # Forward Pass
                text_logits, class_logits = self.model(src, tgt_in)

                # Reshape
                text_logits_flat = text_logits.view(-1, text_logits.size(-1))
                tgt_out_flat = tgt_out.view(-1)

                # Losses
                loss_text = self.criterion_text(text_logits_flat, tgt_out_flat)
                loss_class = self.criterion_class(class_logits, class_id)
                total_loss = loss_text + (Config.LAMBDA_CLASS_LOSS * loss_class)

                # Metrics
                batch_size = src.size(0)

                # Text Accuracy
                preds_text = torch.argmax(text_logits_flat, dim=1)
                mask = tgt_out_flat != Config.PAD_IDX
                correct_text = (preds_text[mask] == tgt_out_flat[mask]).sum().item()
                total_text = mask.sum().item()
                text_acc = correct_text / total_text if total_text > 0 else 0.0

                # Class Accuracy
                preds_class = torch.argmax(class_logits, dim=1)
                correct_class = (preds_class == class_id).sum().item()
                class_acc = correct_class / batch_size

                # Update Trackers
                loss_tracker.update(total_loss.item(), batch_size)
                text_loss_tracker.update(loss_text.item(), batch_size)
                class_loss_tracker.update(loss_class.item(), batch_size)
                text_acc_tracker.update(text_acc, batch_size)
                class_acc_tracker.update(class_acc, batch_size)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} [Val]   Loss: {loss_tracker.avg} | Text Loss: {text_loss_tracker.avg} | Class Loss: {class_loss_tracker.avg} | Text Acc: {text_acc_tracker.avg} | Class Acc: {class_acc_tracker.avg} | Time: {elapsed:.2f}s"
        )

        return loss_tracker.avg

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Scheduler Step
            if self.scheduler:
                # If using ReduceLROnPlateau, pass val_loss
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                print(
                    f"Validation loss improved from {best_val_loss} to {val_loss}. Saving checkpoint..."
                )
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    Config.MODEL_CHECKPOINT,
                )
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
