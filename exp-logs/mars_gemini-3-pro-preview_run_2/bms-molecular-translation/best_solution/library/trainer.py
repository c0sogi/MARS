import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    AverageMeter,
    calc_levenshtein,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import InChiDataset, CollateFn
from library.model import CNNTransformerCTC
from library.tokenizer import InChiTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, debug=False):
        self.device = torch.device(Config.DEVICE)
        self.tokenizer = InChiTokenizer()
        self.debug = debug

        # Initialize Model
        self.model = CNNTransformerCTC().to(self.device)

        # Loss Function
        # CTC Loss requires log_probs. blank=0 is defined in Config.VOCAB
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def get_dataloader(self, mode, sample_size=None):
        if mode == "train":
            csv_path = Config.TRAIN_CSV
            shuffle = True
        elif mode == "val":
            csv_path = Config.VAL_CSV
            shuffle = False
        elif mode == "test":
            csv_path = Config.TEST_CSV
            shuffle = False
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dataset = InChiDataset(csv_path=csv_path, mode=mode, sample_size=sample_size)

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            collate_fn=CollateFn(),
            pin_memory=True if self.device.type == "cuda" else False,
        )
        return dataloader

    def train_epoch(self, dataloader):
        self.model.train()
        losses = AverageMeter()

        for batch_idx, batch in enumerate(dataloader):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]

            # Forward pass
            # Model output: (N, T, C)
            logits = self.model(images)

            # Prepare for CTC Loss
            # 1. Log Softmax
            log_probs = nn.functional.log_softmax(logits, dim=2)

            # 2. Permute to (T, N, C) for CTCLoss
            log_probs = log_probs.permute(1, 0, 2)

            # 3. Calculate Input Lengths for CTC
            # The CNN encoder downsamples width by 4.
            # input_lengths in batch are original image widths.
            # We need the sequence length output by the CNN.
            T = log_probs.size(0)

            # Scale the original widths by the downsampling factor (4).
            # Ensure at least 1 and at most T.
            valid_input_lengths = torch.clamp(input_lengths // 4, min=1).to(torch.long)
            valid_input_lengths = torch.clamp(valid_input_lengths, max=T)

            # Calculate Loss
            loss = self.criterion(
                log_probs, targets, valid_input_lengths, target_lengths
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.MAX_NORM)

            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate_epoch(self, dataloader):
        self.model.eval()
        losses = AverageMeter()
        levenshtein_scores = AverageMeter()

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device)
                targets = batch["targets"].to(self.device)
                input_lengths = batch["input_lengths"]
                target_lengths = batch["target_lengths"]
                target_texts = batch["target_texts"]

                logits = self.model(images)

                # Loss Calculation
                log_probs = nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)
                T = log_probs.size(0)
                valid_input_lengths = torch.clamp(input_lengths // 4, min=1).to(
                    torch.long
                )
                valid_input_lengths = torch.clamp(valid_input_lengths, max=T)

                loss = self.criterion(
                    log_probs, targets, valid_input_lengths, target_lengths
                )
                losses.update(loss.item(), images.size(0))

                # Metric Calculation
                # Decode predictions
                preds = self.tokenizer.decode_ctc_greedy(logits, batch_first=True)

                # Calculate Levenshtein
                score = calc_levenshtein(preds, target_texts)
                levenshtein_scores.update(score, images.size(0))

        return losses.avg, levenshtein_scores.avg

    def fit(self):
        set_seed(Config.SEED)

        sample_size = 1000 if self.debug else None
        train_loader = self.get_dataloader("train", sample_size=sample_size)
        val_loader = self.get_dataloader("val", sample_size=sample_size)

        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(
            f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
        )

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_score = self.validate_epoch(val_loader)

            self.scheduler.step(val_score)

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Levenshtein: {val_score}"
            )

            # Save Checkpoint
            is_best = val_score < best_score
            if is_best:
                best_score = val_score
                patience_counter = 0
                print(f"New best score: {best_score}")
            else:
                patience_counter += 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_score": best_score,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                is_best,
            )

            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    def predict(self):
        print("Starting prediction...")
        # Load Best Model
        load_checkpoint(Config.MODEL_PATH, self.model)
        self.model.eval()

        test_loader = self.get_dataloader("test")

        results = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["images"].to(self.device)
                image_ids = batch["image_ids"]

                logits = self.model(images)
                preds = self.tokenizer.decode_ctc_greedy(logits, batch_first=True)

                for img_id, pred in zip(image_ids, preds):
                    results.append({"image_id": img_id, "InChI": pred})

        df_submission = pd.DataFrame(results)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
