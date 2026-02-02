import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.data import get_data, get_tokenizer, EssayDataset, SlidingWindowDataset
from library.deberta_model import EssayRegressor


class Stage1Trainer:
    """
    Trainer class for Stage 1: Fine-tuning DeBERTa and extracting embeddings.
    """

    def __init__(self):
        """
        Initialize the trainer with configuration, device, and tokenizer.
        """
        seed_everything()
        self.device = Config.DEVICE
        self.tokenizer = get_tokenizer()

    def train_deberta(self):
        """
        Fine-tunes the DeBERTa model on the training set and validates on the validation set.
        Saves the best model based on Validation MSE.
        """
        print("Starting Stage 1: DeBERTa Fine-tuning...")

        # 1. Load Data
        train_df = get_data(Config.TRAIN_DATA_PATH)
        val_df = get_data(Config.VAL_DATA_PATH)

        # 2. Create Datasets and Dataloaders
        train_dataset = EssayDataset(train_df, self.tokenizer)
        val_dataset = EssayDataset(val_df, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Initialize Model
        model = EssayRegressor().to(self.device)

        # 4. Optimizer and Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_training_steps = (
            len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
        ) * Config.EPOCHS
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        criterion = nn.MSELoss()

        # 5. Training Loop
        best_val_loss = float("inf")
        patience = 0
        early_stopping_patience = 2  # Stop if no improvement for 2 epochs

        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss_sum = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, attention_mask)
                    # Squeeze(1) to match shape (batch_size,)
                    loss = criterion(outputs.squeeze(1), labels)
                    loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), Config.MAX_GRAD_NORM
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss_sum += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

            avg_train_loss = train_loss_sum / len(train_loader)

            # Validation
            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(1), labels)
                    val_loss_sum += loss.item()

            avg_val_loss = val_loss_sum / len(val_loader)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train MSE: {avg_train_loss} - Val MSE: {avg_val_loss}"
            )

            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved to {Config.BEST_MODEL_PATH}")
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        print("Fine-tuning complete.")

    def _get_embedding_path(self, base_path):
        """Helper to handle debug file naming."""
        if Config.DEBUG:
            root, ext = os.path.splitext(base_path)
            return f"{root}_debug{ext}"
        return base_path

    def extract_embeddings(self, load_cached_data=True):
        """
        Generates embeddings for Train, Val, and Test sets using the fine-tuned model
        and a sliding window strategy.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            dict: Dictionary containing 'train', 'val', 'test' embedding arrays.
        """
        print("Starting Stage 1: Embedding Extraction...")

        train_emb_path = self._get_embedding_path(Config.TRAIN_EMBEDDINGS_PATH)
        val_emb_path = self._get_embedding_path(Config.VAL_EMBEDDINGS_PATH)
        test_emb_path = self._get_embedding_path(Config.TEST_EMBEDDINGS_PATH)

        # Check cache
        if load_cached_data:
            if (
                os.path.exists(train_emb_path)
                and os.path.exists(val_emb_path)
                and os.path.exists(test_emb_path)
            ):
                print("Loading embeddings from cache...")
                return {
                    "train": np.load(train_emb_path),
                    "val": np.load(val_emb_path),
                    "test": np.load(test_emb_path),
                }
            else:
                print("Cache not found or incomplete. Computing embeddings...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {Config.BEST_MODEL_PATH}. Run train_deberta() first."
            )

        model = EssayRegressor()
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        model.to(self.device)
        model.eval()

        # Process function
        def process_split(data_path, output_path):
            print(f"Processing {data_path}...")
            df = get_data(data_path)
            dataset = SlidingWindowDataset(df, self.tokenizer)
            # Batch size 1 because each item contains multiple chunks
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            all_embeddings = []

            with torch.no_grad():
                for batch in loader:
                    # Shape: (1, num_chunks, window_size) -> (num_chunks, window_size)
                    input_ids = batch["input_ids"].squeeze(0).to(self.device)
                    attention_mask = batch["attention_mask"].squeeze(0).to(self.device)

                    # Get embeddings for all chunks
                    # Shape: (num_chunks, hidden_size)
                    chunk_embeddings = model(
                        input_ids, attention_mask, return_embedding=True
                    )

                    # Mean pool over chunks to get single essay representation
                    # Shape: (hidden_size)
                    essay_embedding = torch.mean(chunk_embeddings, dim=0)

                    all_embeddings.append(essay_embedding.cpu().numpy())

            # Stack to (num_essays, hidden_size)
            final_embeddings = np.vstack(all_embeddings)

            # Save
            np.save(output_path, final_embeddings)
            return final_embeddings

        # Execute for all splits
        train_embeddings = process_split(Config.TRAIN_DATA_PATH, train_emb_path)
        val_embeddings = process_split(Config.VAL_DATA_PATH, val_emb_path)
        test_embeddings = process_split(Config.TEST_DATA_PATH, test_emb_path)

        print("Embedding extraction complete.")

        return {
            "train": train_embeddings,
            "val": val_embeddings,
            "test": test_embeddings,
        }
