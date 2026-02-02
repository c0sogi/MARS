import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import AverageMeter, levenshtein_distance, save_checkpoint
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms
from library.model import ResNetTCN


class Trainer:
    """
    Trainer class to manage the training and validation loop.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device, tokenizer
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Autoregressive training setup
            # Input: Sequence up to the last token (excluding the prediction target)
            # Target: Sequence shifted by one (predicting the next token)
            inputs = labels[:, :-1]
            targets = labels[:, 1:]

            # Forward pass
            # outputs shape: (Batch, Seq_Len, Vocab_Size)
            outputs = self.model(images, inputs)

            # Reshape for CrossEntropyLoss: (N, C) vs (N)
            vocab_size = outputs.size(2)
            loss = self.criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        lev_distances = AverageMeter()

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                inputs = labels[:, :-1]
                targets = labels[:, 1:]

                # Calculate Validation Loss
                outputs = self.model(images, inputs)
                vocab_size = outputs.size(2)
                loss = self.criterion(
                    outputs.reshape(-1, vocab_size), targets.reshape(-1)
                )
                losses.update(loss.item(), images.size(0))

                # Calculate Levenshtein Distance on a subset (first batch only)
                # Doing this for the whole validation set is too slow during training
                if i == 0:
                    preds = self.batched_predict(images)
                    for j in range(len(preds)):
                        pred_str = preds[j]
                        target_str = self.tokenizer.sequence_to_text(labels[j])
                        dist = levenshtein_distance(pred_str, target_str)
                        lev_distances.update(dist)

        return losses.avg, lev_distances.avg

    def batched_predict(self, images):
        """
        Performs greedy decoding inference on a batch of images.
        """
        self.model.eval()
        batch_size = images.size(0)
        max_len = Config.MAX_LEN

        # 1. Encode images once
        features = self.model.encoder(images)  # (B, 512)

        # 2. Initialize sequences with SOS token
        seqs = torch.full(
            (batch_size, 1),
            self.tokenizer.SOS_IDX,
            dtype=torch.long,
            device=self.device,
        )

        # Track finished sequences to potentially optimize (though we run fixed length here for simplicity)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            # Embed current sequence
            embeddings = self.model.embedding(seqs)  # (B, L, Emb_Dim)

            # Expand image features to match sequence length
            features_repeated = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)

            # Concatenate embeddings and image features
            tcn_input = torch.cat(
                (embeddings, features_repeated), dim=2
            )  # (B, L, Emb+Enc)
            tcn_input = tcn_input.permute(0, 2, 1)  # (B, Channels, L)

            # TCN Forward pass
            output = self.model.tcn(tcn_input)  # (B, Channels, L)

            # Get logits for the last time step
            last_output = output[:, :, -1]  # (B, Channels)
            logits = self.model.decoder(last_output)  # (B, Vocab)

            # Greedy selection
            next_tokens = torch.argmax(logits, dim=1)  # (B,)

            # Append to sequence
            seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)

            # Check for EOS
            is_eos = next_tokens == self.tokenizer.EOS_IDX
            finished = finished | is_eos

            if finished.all():
                break

        # Convert indices to strings
        result_strings = []
        for i in range(batch_size):
            result_strings.append(self.tokenizer.sequence_to_text(seqs[i]))

        return result_strings

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs on device {self.device}...")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)
            val_loss, val_lev = self.validate()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein (Subset): {val_lev:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpointing
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_loss": self.best_loss,
                },
                is_best,
            )

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break


def train(load_cached_data=True):
    """
    Main function to setup and run training.
    """
    # 1. Setup Tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(load_cached_data=load_cached_data)

    # 2. Setup Datasets and Loaders
    train_transform = get_transforms("train")
    val_transform = get_transforms("valid")

    train_dataset = InChiDataset(
        Config.TRAIN_METADATA, tokenizer, transform=train_transform
    )
    val_dataset = InChiDataset(Config.VAL_METADATA, tokenizer, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Setup Model
    device = Config.DEVICE
    model = ResNetTCN(vocab_size=tokenizer.get_vocab_size())
    model = model.to(device)

    # 4. Setup Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    # 5. Initialize Trainer and Fit
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, tokenizer
    )
    trainer.fit(Config.NUM_EPOCHS)

    return trainer


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the best trained model.
    """
    print("Generating submission...")

    # 1. Setup Tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(load_cached_data=load_cached_data)

    # 2. Setup Test Dataset
    test_transform = get_transforms("test")
    test_dataset = InChiDataset(
        Config.TEST_METADATA, tokenizer, transform=test_transform, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    device = Config.DEVICE
    model = ResNetTCN(vocab_size=tokenizer.get_vocab_size())

    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Re-implement batched prediction logic here to be self-contained
            batch_size = images.size(0)
            max_len = Config.MAX_LEN

            features = model.encoder(images)
            seqs = torch.full(
                (batch_size, 1), tokenizer.SOS_IDX, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                embeddings = model.embedding(seqs)
                features_repeated = features.unsqueeze(1).expand(
                    -1, embeddings.size(1), -1
                )
                tcn_input = torch.cat((embeddings, features_repeated), dim=2).permute(
                    0, 2, 1
                )

                output = model.tcn(tcn_input)
                last_output = output[:, :, -1]
                logits = model.decoder(last_output)
                next_tokens = torch.argmax(logits, dim=1)

                seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)
                finished = finished | (next_tokens == tokenizer.EOS_IDX)

                if finished.all():
                    break

            # Decode sequences
            for i in range(batch_size):
                inchi = tokenizer.sequence_to_text(seqs[i])
                results.append({"image_id": image_ids[i], "InChI": inchi})

    # 5. Save Submission
    df = pd.DataFrame(results)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
