import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from library.dataset import StackExchangeDataset, collate_fn
from library.model import PartitionedPoolingDualEncoder
from library.utils import (
    seed_everything,
    AverageMeter,
    save_checkpoint,
    compute_spearman_metric,
)


class Trainer:
    def __init__(
        self,
        model_name="roberta-base",
        epochs=7,
        stop_epoch=3,
        batch_size=8,
        accum_steps=2,
        lr_backbone=2e-5,
        lr_head=1e-3,
        working_dir="./working/idea_22",
        debug=False,
    ):
        """
        Trainer class for the StackExchange task.

        Args:
            model_name (str): Name of the transformer model.
            epochs (int): Total epochs for scheduler calculation (Phantom Scheduling).
            stop_epoch (int): Epoch to stop training at.
            batch_size (int): Physical batch size per device.
            accum_steps (int): Gradient accumulation steps.
            lr_backbone (float): Learning rate for the transformer backbone.
            lr_head (float): Learning rate for the regression head.
            working_dir (str): Directory to save checkpoints and cache.
            debug (bool): Whether to run in debug mode (fewer samples).
        """
        self.model_name = model_name
        self.epochs = epochs
        self.stop_epoch = stop_epoch
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.working_dir = working_dir
        self.debug = debug

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.working_dir, exist_ok=True)
        seed_everything(42)

    def train(self):
        """
        Executes the training pipeline.
        """
        # --- Data Loading ---
        print("Initializing Datasets...")
        train_dataset = StackExchangeDataset(
            split="train",
            tokenizer_name=self.model_name,
            cache_dir=self.working_dir,
            debug=self.debug,
        )
        val_dataset = StackExchangeDataset(
            split="val",
            tokenizer_name=self.model_name,
            cache_dir=self.working_dir,
            debug=self.debug,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # --- Model Setup ---
        print("Initializing Model...")
        model = PartitionedPoolingDualEncoder(model_name=self.model_name)
        model.to(self.device)

        # Differential Learning Rates
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n],
                "lr": self.lr_backbone,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" not in n
                ],
                "lr": self.lr_head,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        # Phantom Scheduling
        num_update_steps_per_epoch = len(train_loader) // self.accum_steps
        max_train_steps = num_update_steps_per_epoch * self.epochs
        num_warmup_steps = int(0.1 * max_train_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        criterion = nn.BCEWithLogitsLoss()
        best_score = -1.0

        # --- Training Loop ---
        print("Starting Training...")
        for epoch in range(1, self.stop_epoch + 1):
            model.train()

            # Head Warmup: Freeze backbone in Epoch 1
            if epoch == 1:
                print("Epoch 1: Freezing Backbone for Head Warmup")
                for param in model.backbone.parameters():
                    param.requires_grad = False
            elif epoch == 2:
                print("Epoch 2: Unfreezing Backbone")
                for param in model.backbone.parameters():
                    param.requires_grad = True

            train_loss = AverageMeter()
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                # Move to device
                input_ids_q = batch["input_ids_q"].to(self.device)
                attention_mask_q = batch["attention_mask_q"].to(self.device)
                title_mask = batch["title_mask"].to(self.device)
                body_mask = batch["body_mask"].to(self.device)
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = model(
                    input_ids_q,
                    attention_mask_q,
                    title_mask,
                    body_mask,
                    input_ids_a,
                    attention_mask_a,
                )

                loss = criterion(logits, labels)

                # Gradient Accumulation
                loss = loss / self.accum_steps
                loss.backward()

                if (step + 1) % self.accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss.update(loss.item() * self.accum_steps, labels.size(0))

            # Validation
            val_score = self.validate(model, val_loader)
            print(
                f"Epoch {epoch} | Train Loss: {train_loss.avg:.6f} | Val Spearman: {val_score}"
            )

            is_best = val_score > best_score
            if is_best:
                best_score = val_score

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
                checkpoint_dir=self.working_dir,
            )

        print(f"Training finished. Best Val Score: {best_score}")

        # --- Inference ---
        print("Generating Submission...")
        self.generate_submission(model)

    def validate(self, model, dataloader):
        """
        Runs validation on the provided dataloader.
        """
        model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids_q = batch["input_ids_q"].to(self.device)
                attention_mask_q = batch["attention_mask_q"].to(self.device)
                title_mask = batch["title_mask"].to(self.device)
                body_mask = batch["body_mask"].to(self.device)
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = model(
                    input_ids_q,
                    attention_mask_q,
                    title_mask,
                    body_mask,
                    input_ids_a,
                    attention_mask_a,
                )

                # Sigmoid for prediction
                probs = torch.sigmoid(logits)

                preds.append(probs.cpu().numpy())
                targets.append(labels.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        return compute_spearman_metric(targets, preds)

    def generate_submission(self, model):
        """
        Generates predictions for the test set using the best model and saves to CSV.
        """
        # Load best model
        best_path = os.path.join(self.working_dir, "best_model.pth")
        if not os.path.exists(best_path):
            print(
                f"Best model not found at {best_path}. Skipping submission generation."
            )
            return

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        test_dataset = StackExchangeDataset(
            split="test",
            tokenizer_name=self.model_name,
            cache_dir=self.working_dir,
            debug=self.debug,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

        all_preds = []
        all_qa_ids = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids_q = batch["input_ids_q"].to(self.device)
                attention_mask_q = batch["attention_mask_q"].to(self.device)
                title_mask = batch["title_mask"].to(self.device)
                body_mask = batch["body_mask"].to(self.device)
                input_ids_a = batch["input_ids_a"].to(self.device)
                attention_mask_a = batch["attention_mask_a"].to(self.device)

                logits = model(
                    input_ids_q,
                    attention_mask_q,
                    title_mask,
                    body_mask,
                    input_ids_a,
                    attention_mask_a,
                )
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())
                all_qa_ids.extend(batch["qa_id"])

        all_preds = np.concatenate(all_preds, axis=0)

        # Create submission DF
        sample_sub = pd.read_csv("./input/sample_submission.csv")
        target_cols = [c for c in sample_sub.columns if c != "qa_id"]

        sub_df = pd.DataFrame(all_preds, columns=target_cols)
        sub_df.insert(0, "qa_id", all_qa_ids)

        # Ensure directory exists
        os.makedirs("./submission", exist_ok=True)
        sub_df.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
