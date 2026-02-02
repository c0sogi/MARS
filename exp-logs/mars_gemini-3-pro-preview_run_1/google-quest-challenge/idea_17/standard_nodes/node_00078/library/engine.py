import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import AverageMeter, compute_spearman_correlation, seed_everything
from library.dataset import load_data, StackExchangeDataset, Collate
from library.model import DistilRoBERTaDualEncoder


class Engine:
    def __init__(self):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.collate_fn = Collate(self.tokenizer)
        seed_everything(Config.SEED)

    def get_optimizer_params(self, model):
        """
        Sets up parameter groups for differential learning rates and weight decay.
        """
        # Separate Head and Backbone parameters
        head_params = list(model.head.named_parameters())
        # Backbone includes q_backbone, a_backbone, and the layer_norm before the head
        backbone_params = (
            list(model.q_backbone.named_parameters())
            + list(model.a_backbone.named_parameters())
            + list(model.layer_norm.named_parameters())
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            # Group 1: Head parameters with weight decay
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.HEAD_LR,
            },
            # Group 2: Head parameters without weight decay
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.HEAD_LR,
            },
            # Group 3: Backbone parameters with weight decay
            {
                "params": [
                    p for n, p in backbone_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
                "lr": Config.BACKBONE_LR,
            },
            # Group 4: Backbone parameters without weight decay
            {
                "params": [
                    p for n, p in backbone_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.BACKBONE_LR,
            },
        ]
        return optimizer_grouped_parameters

    def train_one_epoch(self, model, dataloader, optimizer, scheduler, epoch):
        model.train()
        loss_meter = AverageMeter()

        # Loss function: BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
        criterion = nn.BCEWithLogitsLoss()

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            # Forward pass
            logits = model(
                q_input_ids=batch["q_input_ids"],
                q_attention_mask=batch["q_attention_mask"],
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
            )

            loss = criterion(logits, batch["labels"])

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS

            # Backward pass
            loss.backward()

            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_meter.update(
                loss.item() * Config.GRAD_ACCUM_STEPS, batch["q_input_ids"].size(0)
            )

        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] Train Loss: {loss_meter.avg:.6f}")
        return loss_meter.avg

    def validate(self, model, dataloader):
        model.eval()
        loss_meter = AverageMeter()
        criterion = nn.BCEWithLogitsLoss()

        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in dataloader:
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                logits = model(
                    q_input_ids=batch["q_input_ids"],
                    q_attention_mask=batch["q_attention_mask"],
                    a_input_ids=batch["a_input_ids"],
                    a_attention_mask=batch["a_attention_mask"],
                )

                loss = criterion(logits, batch["labels"])
                loss_meter.update(loss.item(), batch["q_input_ids"].size(0))

                # Apply sigmoid for correlation calculation (though rank correlation is monotonic invariant,
                # probabilities are the expected output format)
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                targets_list.append(batch["labels"].cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        score = compute_spearman_correlation(preds, targets)
        print(f"Validation Loss: {loss_meter.avg:.6f}")
        print(f"Validation Spearman Correlation: {score:.10f}")

        return score

    def predict(self, model_path, test_df):
        print("Starting Prediction...")
        # Load Model
        model = DistilRoBERTaDualEncoder()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        # Create Dataset and Loader
        test_dataset = StackExchangeDataset(test_df, self.tokenizer, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        preds_list = []
        qa_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                q_input_ids = batch["q_input_ids"].to(self.device)
                q_attention_mask = batch["q_attention_mask"].to(self.device)
                a_input_ids = batch["a_input_ids"].to(self.device)
                a_attention_mask = batch["a_attention_mask"].to(self.device)

                logits = model(
                    q_input_ids=q_input_ids,
                    q_attention_mask=q_attention_mask,
                    a_input_ids=a_input_ids,
                    a_attention_mask=a_attention_mask,
                )

                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                preds_list.append(probs.cpu().numpy())
                qa_ids_list.append(batch["qa_id"].numpy())

        all_preds = np.concatenate(preds_list, axis=0)
        all_qa_ids = np.concatenate(qa_ids_list, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)
        submission_df.insert(0, "qa_id", all_qa_ids)

        # Save
        Config.create_dirs()
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())

    def run(self):
        # 1. Load Data
        train_df, val_df, test_df = load_data(load_cached_data=True)

        # 2. Prepare Datasets and Loaders
        train_dataset = StackExchangeDataset(train_df, self.tokenizer, is_test=False)
        val_dataset = StackExchangeDataset(val_df, self.tokenizer, is_test=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        # 3. Initialize Model
        model = DistilRoBERTaDualEncoder()
        model.to(self.device)

        # 4. Optimizer and Scheduler
        optimizer_parameters = self.get_optimizer_params(model)
        optimizer = torch.optim.AdamW(optimizer_parameters)

        num_train_steps = len(train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # 5. Training Loop
        best_score = -1.0
        patience = 0
        early_stopping_limit = 2  # Stop if no improvement for 2 epochs

        Config.create_dirs()

        for epoch in range(Config.EPOCHS):
            # Schedule Logic
            if epoch == 0:
                print(f"\nEpoch {epoch+1}: Freezing backbone. Training Head only.")
                model.freeze_backbone()
            elif epoch == 1:
                print(f"\nEpoch {epoch+1}: Unfreezing backbone. Full Fine-tuning.")
                model.unfreeze_backbone()
            else:
                print(f"\nEpoch {epoch+1}: Full Fine-tuning.")

            # Train
            self.train_one_epoch(model, train_loader, optimizer, scheduler, epoch)

            # Validate
            val_score = self.validate(model, val_loader)

            # Save Best Model
            if val_score > best_score:
                print(
                    f"Score Improved ({best_score:.6f} -> {val_score:.6f}). Saving Model..."
                )
                best_score = val_score
                torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
                patience = 0
            else:
                print(f"Score did not improve (Best: {best_score:.6f}).")
                patience += 1
                if patience >= early_stopping_limit:
                    print("Early stopping triggered.")
                    break

        # 6. Prediction
        self.predict(Config.BEST_MODEL_PATH, test_df)
