import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, compute_levenshtein_distance, decode_predictions
from library.data_loader import GestureDataset
from library.model import CGRNet

# Global cache for dense labels to be used in collate_fn
DENSE_LABEL_CACHE = {}


def load_dense_labels(df, cache_path, load_cached_data=True):
    """
    Loads or computes frame-wise dense labels for the given DataFrame.
    Stores the result in the global DENSE_LABEL_CACHE.
    """
    global DENSE_LABEL_CACHE

    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading dense labels from {cache_path}...")
            loaded = np.load(cache_path, allow_pickle=True).item()
            DENSE_LABEL_CACHE.update(loaded)
            return
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Computing dense labels from MAT files...")
    new_cache = {}

    # Filter valid rows
    valid_df = df[df["data_path"].notna()]

    for _, row in tqdm(
        valid_df.iterrows(), total=len(valid_df), desc="Processing Labels"
    ):
        sample_id = row["sample_id"]
        rel_path = row["data_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Default: all background
        # We need num_frames. If not in df, read from MAT.
        # Ideally we read MAT to get accurate frame-wise annotations.
        try:
            mat = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
            if "Video" not in mat:
                continue
            video = mat["Video"]
            num_frames = getattr(video, "NumFrames", 0)

            # Initialize with Background Class (0)
            dense = np.zeros(num_frames, dtype=np.int64)

            if hasattr(video, "Labels"):
                labels_data = video.Labels
                if not isinstance(labels_data, np.ndarray):
                    labels_data = [labels_data]
                elif labels_data.size == 1:
                    labels_data = [labels_data.item()]

                for l in labels_data:
                    if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                        name = l.Name
                        # MATLAB indices are 1-based, convert to 0-based
                        # Range is inclusive in MATLAB usually
                        start = max(0, int(l.Begin) - 1)
                        end = min(num_frames, int(l.End))

                        if name in Config.LABEL_MAP:
                            lid = Config.LABEL_MAP[name]
                            dense[start:end] = lid

            new_cache[sample_id] = dense

        except Exception as e:
            # print(f"Error parsing {sample_id}: {e}")
            pass

    # Update global cache and save
    DENSE_LABEL_CACHE.update(new_cache)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, DENSE_LABEL_CACHE)
    print("Dense labels cached.")


def robust_collate_fn(batch):
    """
    Custom collate function that handles missing dense_labels in the batch
    by looking them up in the global DENSE_LABEL_CACHE.
    """
    # Sort by length (descending)
    batch.sort(key=lambda x: x["skeleton"].shape[0], reverse=True)

    skeletons = [x["skeleton"] for x in batch]
    audios = [x["audio"] for x in batch]
    labels_list = [x["labels"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]

    lengths = torch.tensor([s.shape[0] for s in skeletons], dtype=torch.long)

    # Pad inputs
    skeletons_padded = pad_sequence(skeletons, batch_first=True, padding_value=0.0)
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

    # Create Mask (True for valid positions, False for padding)
    max_len = skeletons_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    # Handle Dense Labels
    dense_labels_list = []
    for i, sid in enumerate(sample_ids):
        # Look up in cache
        if sid in DENSE_LABEL_CACHE:
            d_lbl = torch.tensor(DENSE_LABEL_CACHE[sid], dtype=torch.long)
            # Ensure length matches skeleton (augmentations/processing might differ slightly or file mismatch)
            # Truncate or pad to match current skeleton length
            target_len = lengths[i].item()
            if d_lbl.shape[0] > target_len:
                d_lbl = d_lbl[:target_len]
            elif d_lbl.shape[0] < target_len:
                pad_amt = target_len - d_lbl.shape[0]
                d_lbl = torch.cat([d_lbl, torch.zeros(pad_amt, dtype=torch.long)])
            dense_labels_list.append(d_lbl)
        else:
            # Fallback (e.g., for test set or missing cache)
            # Create zeros
            dense_labels_list.append(torch.zeros(lengths[i].item(), dtype=torch.long))

    dense_labels_padded = pad_sequence(
        dense_labels_list, batch_first=True, padding_value=Config.BACKGROUND_CLASS_ID
    )

    return {
        "skeleton": skeletons_padded,
        "audio": audios_padded,
        "dense_labels": dense_labels_padded,
        "seq_labels": labels_list,
        "lengths": lengths,
        "mask": mask,
        "sample_ids": sample_ids,
    }


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss with Class Weighting
        # Background (0) gets 0.5, others 1.0
        weights = torch.ones(Config.NUM_CLASSES, device=Config.DEVICE)
        weights[Config.BACKGROUND_CLASS_ID] = Config.BG_WEIGHT
        self.criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX
        )

        self.best_metric = float("inf")
        self.patience = 10
        self.counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            skeleton = batch["skeleton"].to(Config.DEVICE)
            audio = batch["audio"].to(Config.DEVICE)
            mask = batch["mask"].to(Config.DEVICE)
            targets = batch["dense_labels"].to(Config.DEVICE)

            self.optimizer.zero_grad()

            logits = self.model(skeleton, audio, mask)

            # Flatten for CrossEntropy: (B*T, C) vs (B*T)
            # Only calculate loss on valid frames (using mask)
            active_logits = logits[mask]
            active_targets = targets[mask]

            loss = self.criterion(active_logits, active_targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch in self.val_loader:
                skeleton = batch["skeleton"].to(Config.DEVICE)
                audio = batch["audio"].to(Config.DEVICE)
                mask = batch["mask"].to(Config.DEVICE)

                logits = self.model(skeleton, audio, mask)  # (B, T, C)

                # Decode
                for i in range(logits.size(0)):
                    # Slice valid length
                    valid_len = batch["lengths"][i]
                    seq_logits = logits[i, :valid_len, :]

                    pred_seq = decode_predictions(
                        seq_logits, threshold=5, bg_class=Config.BACKGROUND_CLASS_ID
                    )
                    all_preds.append(pred_seq)

                    # Truth
                    truth_seq = batch["seq_labels"][i].tolist()
                    all_truths.append(truth_seq)

        # Compute Metric
        score = compute_levenshtein_distance(all_preds, all_truths)
        return score

    def fit(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Levenshtein: {val_score}"
            )

            # Checkpoint
            if val_score < self.best_metric:
                self.best_metric = val_score
                self.counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print("  -> New best model saved.")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training finished. Best Val Score: {self.best_metric}")

    def predict(self):
        print("Generating predictions for test set...")
        # Load best model
        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
        self.model.eval()

        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                skeleton = batch["skeleton"].to(Config.DEVICE)
                audio = batch["audio"].to(Config.DEVICE)
                mask = batch["mask"].to(Config.DEVICE)
                sample_ids = batch["sample_ids"]

                logits = self.model(skeleton, audio, mask)

                for i in range(logits.size(0)):
                    valid_len = batch["lengths"][i]
                    seq_logits = logits[i, :valid_len, :]
                    pred_seq = decode_predictions(
                        seq_logits, threshold=5, bg_class=Config.BACKGROUND_CLASS_ID
                    )

                    # Format: SessionID,1,2,3
                    pred_str = ",".join(map(str, pred_seq))
                    results.append(f"{sample_ids[i]},{pred_str}")

        # Save submission
        with open(Config.SUBMISSION_PATH, "w") as f:
            for line in results:
                f.write(line + "\n")
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_module():
    set_seed(Config.SEED)

    # 1. Prepare Data
    # Pre-compute dense labels for Train and Val
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    cache_path = os.path.join(Config.WORK_DIR, "dense_labels_cache.npy")
    load_dense_labels(combined_df, cache_path, load_cached_data=True)

    # Datasets
    train_dataset = GestureDataset(split="train", load_cached_data=True)
    val_dataset = GestureDataset(split="val", load_cached_data=True)
    test_dataset = GestureDataset(split="test", load_cached_data=True)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=robust_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=robust_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=robust_collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model
    model = CGRNet()

    # 3. Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # 4. Fit
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # 5. Predict
    trainer.predict()


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing.
    # The prompt asks for module implementation only.
    pass
