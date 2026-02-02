import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForTokenClassification
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.data_factory import get_dataloader


class QATrainer:
    """
    Manages the supervised fine-tuning of the Question Answering model.
    Handles model initialization (from TAPT or Base), training, validation,
    and saving the best state.
    """

    def __init__(self, seed):
        self.seed = seed
        self.device = torch.device(Config.DEVICE)
        self.best_val_loss = float("inf")
        self.model_save_path = os.path.join(
            Config.QA_MODEL_DIR, f"model_seed_{seed}.pt"
        )

    def get_model(self):
        """
        Initializes the model. Prioritizes the TAPT-finetuned weights.
        """
        # Check if TAPT model exists
        tapt_weights_path = Config.TAPT_MODEL_DIR
        has_tapt = os.path.exists(
            os.path.join(tapt_weights_path, "config.json")
        ) and os.path.exists(os.path.join(tapt_weights_path, "model.safetensors"))

        model_path = tapt_weights_path if has_tapt else Config.BASE_MODEL_NAME

        if has_tapt:
            print(
                f"[Seed {self.seed}] Initializing model from TAPT weights: {model_path}"
            )
        else:
            print(
                f"[Seed {self.seed}] TAPT weights not found. Initializing from base: {model_path}"
            )

        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=Config.NUM_LABELS,
        )
        return model.to(self.device)

    def train_one_epoch(self, model, dataloader, optimizer):
        """
        Runs one epoch of training.
        """
        model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            # Filter batch for model inputs (remove metadata like offset_mapping, example_id)
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def validate(self, model, dataloader):
        """
        Runs validation loop and returns average loss.
        """
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask", "labels"]
                }

                outputs = model(**inputs)
                loss = outputs.loss

                total_loss += loss.item()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def run(self, load_cached_data=True):
        """
        Executes the full training pipeline for this seed.
        """
        seed_everything(self.seed)
        print(f"\nStarting QA Training for Seed {self.seed}")

        # 1. Load Data
        train_loader = get_dataloader(
            mode="train",
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            load_cached_data=load_cached_data,
        )
        val_loader = get_dataloader(
            mode="val",
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            load_cached_data=load_cached_data,
        )

        # 2. Initialize Model & Optimizer
        model = self.get_model()
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 3. Training Loop
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_one_epoch(model, train_loader, optimizer)
            val_loss = self.validate(model, val_loader)

            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f}"
            )

            # 4. Save Best Model
            if val_loss < self.best_val_loss:
                print(
                    f"Validation Loss Improved ({self.best_val_loss:.8f} -> {val_loss:.8f}). Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(model.state_dict(), self.model_save_path)
            else:
                print("Validation Loss did not improve.")

        print(
            f"Finished training for Seed {self.seed}. Best Val Loss: {self.best_val_loss:.8f}"
        )


def run_training(load_cached_data=True):
    """
    Orchestrates training for all seeds defined in Config.
    """
    # Ensure output directory exists
    os.makedirs(Config.QA_MODEL_DIR, exist_ok=True)

    for seed in Config.SEEDS:
        trainer = QATrainer(seed)
        trainer.run(load_cached_data=load_cached_data)

    print("\nAll training runs completed successfully.")
