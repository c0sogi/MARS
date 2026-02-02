import torch
import torch.nn as nn
import torch.optim as optim
import os
from library.config import Config


class TrainEngine:
    def __init__(self, model, device=None):
        self.model = model
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Loss Functions
        # Sentence ranking: Binary Cross Entropy (scores are Cosine Sim [0, 1] due to ReLU encoder)
        self.sent_criterion = nn.BCELoss()
        # Yes/No classification: Categorical Cross Entropy (logits)
        self.yn_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_sent_loss = 0.0
        total_yn_loss = 0.0
        count = 0

        for batch in train_loader:
            # Move data to device
            questions = batch["questions"].to(self.device)
            sentences = batch["sentences"].to(self.device)
            labels = batch["labels"].to(self.device)
            yes_no_targets = batch["yes_no"].to(self.device)
            doc_lengths = batch[
                "doc_lengths"
            ]  # List, handled inside model or converted there

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # scores: (total_sentences,)
            # yn_logits: (batch_size, 3)
            scores, yn_logits = self.model(questions, sentences, doc_lengths)

            # Calculate Loss
            # Ensure scores are float for BCELoss
            loss_sent = self.sent_criterion(scores, labels.float())
            loss_yn = self.yn_criterion(yn_logits, yes_no_targets)

            # Combined loss
            loss = loss_sent + loss_yn

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            batch_size = questions.size(0)
            total_loss += loss.item() * batch_size
            total_sent_loss += loss_sent.item() * batch_size
            total_yn_loss += loss_yn.item() * batch_size
            count += batch_size

        avg_loss = total_loss / count if count > 0 else 0.0
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct_yn = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                questions = batch["questions"].to(self.device)
                sentences = batch["sentences"].to(self.device)
                labels = batch["labels"].to(self.device)
                yes_no_targets = batch["yes_no"].to(self.device)
                doc_lengths = batch["doc_lengths"]

                scores, yn_logits = self.model(questions, sentences, doc_lengths)

                loss_sent = self.sent_criterion(scores, labels.float())
                loss_yn = self.yn_criterion(yn_logits, yes_no_targets)
                loss = loss_sent + loss_yn

                total_loss += loss.item() * questions.size(0)

                # Yes/No Accuracy
                preds_yn = torch.argmax(yn_logits, dim=1)
                correct_yn += (preds_yn == yes_no_targets).sum().item()

                total_samples += questions.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        yn_accuracy = correct_yn / total_samples if total_samples > 0 else 0.0

        return avg_loss, yn_accuracy

    def train(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS):
        print(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            val_loss, val_yn_acc = self.validate(val_loader)

            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss: {val_loss}")
            print(f"  Val Yes/No Accuracy: {val_yn_acc}")

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
                print(
                    f"  Validation loss improved. Model saved to {Config.MODEL_CHECKPOINT_PATH}"
                )
            else:
                patience_counter += 1
                print(
                    f"  Validation loss did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")
