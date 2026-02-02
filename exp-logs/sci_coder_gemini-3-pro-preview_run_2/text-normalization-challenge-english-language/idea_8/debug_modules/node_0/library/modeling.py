import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AdamW,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
import numpy as np
from tqdm.auto import tqdm
from library.config import cfg

# Suppress HF warnings
hf_logging.set_verbosity_error()


class RouterModel:
    def __init__(self, model_path=None):
        self.device = torch.device(cfg.DEVICE)

        # Load Pretrained Model
        model_name = model_path if model_path else cfg.ROUTER_MODEL_NAME
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=cfg.NUM_CLASSES,
            id2label=cfg.ID2CLASS,
            label2id=cfg.CLASS2ID,
        )
        self.model.to(self.device)

    def train(self, train_dataset, val_dataset):
        print(f"Initializing Router Training (DeBERTa-v3)...")

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.ROUTER_BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.ROUTER_BATCH_SIZE, shuffle=False, num_workers=2
        )

        optimizer = AdamW(self.model.parameters(), lr=cfg.ROUTER_LEARNING_RATE)

        num_training_steps = len(train_loader) * cfg.ROUTER_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(cfg.ROUTER_EPOCHS):
            # Training Loop
            self.model.train()
            train_loss = 0

            # Use tqdm for progress tracking if interactive, else silent or simple print
            # Using simple print to avoid clutter in logs
            print(f"Epoch {epoch+1}/{cfg.ROUTER_EPOCHS}")

            for batch in tqdm(train_loader, desc="Router Train"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Loop
            val_acc = self.evaluate(val_loader)
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Acc:    {val_acc:.10f}")

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model("router_best")
                print("  -> Best model saved!")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{cfg.ROUTER_PATIENCE}"
                )

            if patience_counter >= cfg.ROUTER_PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model for future use
        self.load_model("router_best")

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Router Eval"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]

                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)

                # Mask for active labels (not -100)
                active_mask = labels != -100

                active_preds = predictions[active_mask]
                active_labels = labels[active_mask]

                correct += (active_preds == active_labels).sum().item()
                total += active_labels.numel()

        return correct / total if total > 0 else 0.0

    def predict(self, dataset):
        """
        Runs inference on the dataset.
        Returns a list of lists (predicted class IDs per sentence).
        """
        self.model.eval()
        dataloader = DataLoader(
            dataset, batch_size=cfg.ROUTER_BATCH_SIZE * 2, shuffle=False, num_workers=2
        )

        all_preds = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Router Predict"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                # (Batch, Seq_Len)
                predictions = torch.argmax(outputs.logits, dim=-1)

                # We need to map these back to the original word-level structure.
                # However, the dataset provided here is tokenized.
                # The alignment logic happens outside or we return the raw token preds.
                # Given the logic in data_utils, we just return the raw predictions here
                # and let the caller align them using word_ids if necessary.
                # But wait, for the test set, we need to align carefully.
                # To simplify, we return the list of lists of IDs,
                # but we must handle the subword collapsing in the main pipeline or here.
                # Since `data_utils.process_router_data` aligns labels to the first subword,
                # we should extract predictions for the first subword of each token.
                # BUT: The input `dataset` here doesn't have `word_ids` easily accessible
                # unless we passed the raw encoding.
                # Simplification: We return the full subword predictions.
                # The calling code (inference pipeline) must map subwords to words.

                all_preds.extend(predictions.cpu().numpy().tolist())

        return all_preds

    def save_model(self, name):
        path = os.path.join(cfg.CHECKPOINT_DIR, name)
        self.model.save_pretrained(path)

    def load_model(self, name):
        path = os.path.join(cfg.CHECKPOINT_DIR, name)
        if os.path.exists(path):
            self.model = AutoModelForTokenClassification.from_pretrained(path)
            self.model.to(self.device)
            print(f"Loaded Router model from {path}")
        else:
            print(f"Warning: Checkpoint {path} not found.")


class GeneratorModel:
    def __init__(self, model_path=None):
        self.device = torch.device(cfg.DEVICE)

        model_name = model_path if model_path else cfg.GENERATOR_MODEL_NAME
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def train(self, train_dataset, val_dataset):
        print(f"Initializing Generator Training (ByT5)...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.GENERATOR_BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.GENERATOR_BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )

        optimizer = AdamW(self.model.parameters(), lr=cfg.GENERATOR_LEARNING_RATE)

        # Simple linear scheduler
        num_training_steps = len(train_loader) * cfg.GENERATOR_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * num_training_steps),
            num_training_steps=num_training_steps,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg.GENERATOR_EPOCHS):
            self.model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc="Generator Train"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss = self.evaluate_loss(val_loader)
            print(f"Epoch {epoch+1}/{cfg.GENERATOR_EPOCHS}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.10f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("generator_best")
                print("  -> Best model saved!")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{cfg.GENERATOR_PATIENCE}"
                )

            if patience_counter >= cfg.GENERATOR_PATIENCE:
                print("Early stopping triggered.")
                break

        self.load_model("generator_best")

    def evaluate_loss(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generator Eval"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        return total_loss / len(dataloader)

    def predict(self, dataset):
        """
        Generates text for the inputs in the dataset.
        """
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.GENERATOR_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=2,
        )

        generated_texts = []

        # We need the tokenizer to decode
        from library.data_utils import get_generator_tokenizer

        tokenizer = get_generator_tokenizer()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generator Predict"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=cfg.MAX_LENGTH_GENERATOR,
                    num_beams=1,  # Greedy decoding is usually sufficient and faster
                    early_stopping=True,
                )

                decoded = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                generated_texts.extend(decoded)

        return generated_texts

    def save_model(self, name):
        path = os.path.join(cfg.CHECKPOINT_DIR, name)
        self.model.save_pretrained(path)

    def load_model(self, name):
        path = os.path.join(cfg.CHECKPOINT_DIR, name)
        if os.path.exists(path):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
            self.model.to(self.device)
            print(f"Loaded Generator model from {path}")
        else:
            print(f"Warning: Checkpoint {path} not found.")
