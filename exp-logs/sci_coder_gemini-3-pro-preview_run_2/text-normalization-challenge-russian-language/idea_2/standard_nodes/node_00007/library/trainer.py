import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.utils import ensure_dir
from library.model import ContextAwareTransformer
from library.dataset import DigitSeq2SeqDataset
from library.vocab import get_tokenizer


class Trainer:
    """
    Trainer class to manage the training and evaluation of the Seq2Seq model.
    """

    def __init__(self, tokenizer=None):
        """
        Initializes the Trainer with model, optimizer, criterion, and scheduler.

        Args:
            tokenizer (CharTokenizer, optional): Tokenizer instance. If None, loads from cache.
        """
        self.device = Config.DEVICE

        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = get_tokenizer(load_cached_data=True)
        else:
            self.tokenizer = tokenizer

        self.vocab_size = len(self.tokenizer.token2idx)

        # Initialize Model
        self.model = ContextAwareTransformer(
            vocab_size=self.vocab_size,
            embed_dim=Config.EMBED_DIM,
            n_heads=Config.N_HEADS,
            hidden_dim=Config.HIDDEN_DIM,
            n_layers=Config.N_LAYERS,
            dropout=Config.DROPOUT,
            pad_idx=Config.PAD_IDX,
            device=self.device,
        )

        # Initialize Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Initialize Criterion (Loss Function)
        # ignore_index ensures padding tokens don't contribute to loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

        # Initialize Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): The training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0

        for batch in dataloader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)

            # Target Input: [SOS, ..., token_n]
            # Target Output: [token_1, ..., EOS]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(src, tgt_input)

            # Reshape for loss calculation
            # output: [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)

            # tgt_output: [Batch, Seq] -> [Batch*Seq]
            tgt_output = tgt_output.reshape(-1)

            loss = self.criterion(output, tgt_output)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.

        Args:
            dataloader (DataLoader): The validation data loader.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = self.model(src, tgt_input)

                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                tgt_output = tgt_output.reshape(-1)

                loss = self.criterion(output, tgt_output)
                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def fit(self, train_dataset=None, val_dataset=None, load_cached_data=True):
        """
        Runs the full training loop with early stopping.

        Args:
            train_dataset (Dataset, optional): Training dataset. If None, created from config.
            val_dataset (Dataset, optional): Validation dataset. If None, created from config.
            load_cached_data (bool): Whether to load processed data from cache.

        Returns:
            nn.Module: The trained model.
        """
        set_seed(Config.SEED)
        print(f"Training on device: {self.device}")

        # Initialize Datasets if not provided
        if train_dataset is None:
            train_dataset = DigitSeq2SeqDataset(
                mode="train",
                tokenizer=self.tokenizer,
                load_cached_data=load_cached_data,
                debug=Config.DEBUG,
            )

        if val_dataset is None:
            val_dataset = DigitSeq2SeqDataset(
                mode="val",
                tokenizer=self.tokenizer,
                load_cached_data=load_cached_data,
                debug=Config.DEBUG,
            )

        # Initialize DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting training...")
        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            # Update Learning Rate
            self.scheduler.step(val_loss)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {train_loss}")
            print(f"\t Val. Loss: {val_loss}")

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ensure_dir(Config.MODEL_CHECKPOINT)
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                print(f"\tNew best model saved to {Config.MODEL_CHECKPOINT}")
            else:
                patience_counter += 1
                print(
                    f"\tNo improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        return self.model
