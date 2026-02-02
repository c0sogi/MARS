import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.signal import medfilt

from library import config, utils, data_loader, model


class Trainer:
    def __init__(self):
        utils.set_seed()
        self.device = config.DEVICE

        # Initialize Model
        self.model = model.CGRNet().to(self.device)

        # Loss Function
        # Weighted Cross Entropy with Label Smoothing
        class_weights = config.get_class_weights(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING,
            ignore_index=-1,  # Safety for padding if needed, though mask handles it
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler placeholder (initialized in fit)
        self.scheduler = None

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for skeletons, audios, labels, mask in loader:
            skeletons = skeletons.to(self.device)
            audios = audios.to(self.device)
            labels = labels.to(self.device)

            # Lengths for packing (derived from mask or passed explicitly if loader changed)
            # In data_loader.collate_fn, mask is created based on lengths.
            # We can recover lengths from the mask sum.
            lengths = mask.sum(dim=1).cpu()

            self.optimizer.zero_grad()

            # Forward Pass
            logits = self.model(skeletons, audios, lengths)  # (B, T, C)

            # Flatten for Loss Calculation
            # We only calculate loss on valid frames (masked)
            # logits: (B, T, C) -> (B*T, C)
            # labels: (B, T) -> (B*T)

            # Using the mask to select valid elements
            active_logits = logits[mask]
            active_labels = labels[mask]

            loss = self.criterion(active_logits, active_labels)

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds_seq = []
        all_targets_seq = []

        with torch.no_grad():
            for skeletons, audios, labels, mask in loader:
                skeletons = skeletons.to(self.device)
                audios = audios.to(self.device)
                labels = labels.to(self.device)
                lengths = mask.sum(dim=1).cpu()

                # Forward
                logits = self.model(skeletons, audios, lengths)

                # Loss calculation
                active_logits = logits[mask]
                active_labels = labels[mask]
                loss = self.criterion(active_logits, active_labels)
                total_loss += loss.item()
                num_batches += 1

                # Decoding for Metric
                # Apply Softmax
                probs = torch.softmax(logits, dim=2)
                preds = torch.argmax(probs, dim=2)  # (B, T)

                # Convert to CPU numpy
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                mask_np = mask.cpu().numpy()

                batch_size = preds_np.shape[0]
                for i in range(batch_size):
                    # Get valid length
                    length = int(lengths[i].item())

                    # Extract valid frames
                    p_seq = preds_np[i, :length]
                    t_seq = labels_np[i, :length]

                    # Apply Median Filter (Window=5)
                    # Must be odd. Handles noise in frame-wise predictions.
                    if len(p_seq) >= 5:
                        p_seq = medfilt(p_seq, kernel_size=5)

                    # RLE Decode
                    decoded_pred = utils.rle_decode(
                        p_seq, background_label=config.BACKGROUND_LABEL, min_duration=5
                    )

                    decoded_target = utils.rle_decode(
                        t_seq,
                        background_label=config.BACKGROUND_LABEL,
                        min_duration=1,  # Ground truth shouldn't be filtered aggressively
                    )

                    all_preds_seq.append(decoded_pred)
                    all_targets_seq.append(decoded_target)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        ler = utils.compute_levenshtein_ratio(all_preds_seq, all_targets_seq)

        return avg_loss, ler

    def fit(self, limit=None):
        print("Initializing Datasets...")
        train_dataset = data_loader.GestureDataset(
            config.TRAIN_METADATA_PATH, mode="train", limit=limit
        )
        val_dataset = data_loader.GestureDataset(
            config.VAL_METADATA_PATH, mode="val", limit=limit
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=data_loader.collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=data_loader.collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS
        )

        best_ler = float("inf")
        patience = 10
        patience_counter = 0

        print(f"Starting training on {config.DEVICE} for {config.NUM_EPOCHS} epochs...")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_ler = self.validate(val_loader)

            self.scheduler.step()

            print(
                f"Epoch {epoch}: Train Loss={train_loss}, Val Loss={val_loss}, Val LER={val_ler}"
            )

            # Checkpoint based on LER
            if val_ler < best_ler:
                best_ler = val_ler
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training completed. Best Val LER: {best_ler}")

    def predict(self):
        print("Loading best model for inference...")
        model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            print("No checkpoint found. Skipping prediction.")
            return

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        test_dataset = data_loader.GestureDataset(
            config.TEST_METADATA_PATH, mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=data_loader.collate_fn,
            num_workers=2,
        )

        predictions = {}
        # We need sample IDs to map predictions.
        # The dataset is sequential, so we can iterate sample_ids from the dataset object.
        sample_ids = test_dataset.sample_ids
        current_idx = 0

        with torch.no_grad():
            for skeletons, audios, _, mask in test_loader:
                skeletons = skeletons.to(self.device)
                audios = audios.to(self.device)
                lengths = mask.sum(dim=1).cpu()

                logits = self.model(skeletons, audios, lengths)
                preds = torch.argmax(logits, dim=2).cpu().numpy()

                batch_size = preds.shape[0]
                for i in range(batch_size):
                    length = int(lengths[i].item())
                    p_seq = preds[i, :length]

                    # Median Filter
                    if len(p_seq) >= 5:
                        p_seq = medfilt(p_seq, kernel_size=5)

                    # Decode
                    decoded_seq = utils.rle_decode(
                        p_seq, background_label=config.BACKGROUND_LABEL, min_duration=5
                    )

                    sid = sample_ids[current_idx]
                    predictions[sid] = decoded_seq
                    current_idx += 1

        output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        utils.save_submission(predictions, output_path)
        print(f"Submission saved to {output_path}")
