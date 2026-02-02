import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.models import SiameseRanker, ConditionalReader
from library.data_processing import get_data_loaders


class Trainer:
    def __init__(self, load_cached_data=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # Ensure working directory exists for model checkpoints
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Load data loaders and vocabulary
        self.loaders = get_data_loaders(load_cached_data=load_cached_data)
        self.vocab = self.loaders["vocab"]

    def train_ranker_epoch(self, model, optimizer, criterion):
        model.train()
        total_loss = 0.0

        for batch in self.loaders["ranker_train"]:
            q_input = batch["q_input"].to(self.device)
            ctx_input = batch["ctx_input"].to(self.device)
            label = batch["label"].to(self.device)

            # Convert 0/1 labels to -1/1 for CosineEmbeddingLoss
            # Label 1 (Positive) -> Target 1
            # Label 0 (Negative) -> Target -1
            target = label * 2 - 1

            optimizer.zero_grad()

            # Get embeddings from the Siamese network
            q_vec = model.forward_one(q_input)
            ctx_vec = model.forward_one(ctx_input)

            loss = criterion(q_vec, ctx_vec, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.loaders["ranker_train"])

    def validate_ranker(self, model, criterion):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.loaders["ranker_val"]:
                q_input = batch["q_input"].to(self.device)
                ctx_input = batch["ctx_input"].to(self.device)
                label = batch["label"].to(self.device)

                target = label * 2 - 1

                q_vec = model.forward_one(q_input)
                ctx_vec = model.forward_one(ctx_input)

                loss = criterion(q_vec, ctx_vec, target)
                total_loss += loss.item()

        return total_loss / len(self.loaders["ranker_val"])

    def train_ranker(self):
        print("\nStarting Ranker Training...")
        # Initialize model with correct vocab size
        model = SiameseRanker(vocab_size=len(self.vocab)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        # Margin for dissimilar pairs in CosineEmbeddingLoss
        criterion = nn.CosineEmbeddingLoss(margin=0.5)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_ranker_epoch(model, optimizer, criterion)
            val_loss = self.validate_ranker(model, criterion)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Ranker Train Loss: {train_loss} | Ranker Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
                print(f"New best Ranker model saved to {Config.RANKER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered for Ranker at epoch {epoch+1}")
                    break

    def train_reader_epoch(self, model, optimizer, criterion):
        model.train()
        total_loss = 0.0

        for batch in self.loaders["reader_train"]:
            q_input = batch["q_input"].to(self.device)
            ctx_input = batch["ctx_input"].to(self.device)
            start_target = batch["start_target"].to(self.device)
            end_target = batch["end_target"].to(self.device)

            optimizer.zero_grad()

            start_logits, end_logits = model(q_input, ctx_input)

            # Compute loss for both start and end tokens
            loss_start = criterion(start_logits, start_target)
            loss_end = criterion(end_logits, end_target)
            loss = loss_start + loss_end

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.loaders["reader_train"])

    def validate_reader(self, model, criterion):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.loaders["reader_val"]:
                q_input = batch["q_input"].to(self.device)
                ctx_input = batch["ctx_input"].to(self.device)
                start_target = batch["start_target"].to(self.device)
                end_target = batch["end_target"].to(self.device)

                start_logits, end_logits = model(q_input, ctx_input)

                loss_start = criterion(start_logits, start_target)
                loss_end = criterion(end_logits, end_target)
                loss = loss_start + loss_end

                total_loss += loss.item()

        return total_loss / len(self.loaders["reader_val"])

    def train_reader(self):
        print("\nStarting Reader Training...")
        model = ConditionalReader(vocab_size=len(self.vocab)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_reader_epoch(model, optimizer, criterion)
            val_loss = self.validate_reader(model, criterion)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Reader Train Loss: {train_loss} | Reader Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.READER_MODEL_PATH)
                print(f"New best Reader model saved to {Config.READER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered for Reader at epoch {epoch+1}")
                    break


def run_training(load_cached_data=True):
    trainer = Trainer(load_cached_data=load_cached_data)
    trainer.train_ranker()
    trainer.train_reader()
