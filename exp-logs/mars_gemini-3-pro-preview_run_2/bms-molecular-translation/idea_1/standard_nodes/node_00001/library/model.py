import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import ChemicalDataset
from library.utils import AverageMeter, compute_levenshtein
from library.tokenizer import Tokenizer


# Set fixed seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for InChI prediction.

    Architecture:
    1. CNN Backbone: Extracts feature sequence from images.
       - Designed to downsample height aggressively (to 1) while preserving width
         to maintain sequence resolution for long InChI strings.
    2. Bidirectional LSTM: Captures sequence dependencies.
    3. Linear Projection: Maps to character vocabulary.
    """

    def __init__(self):
        super(CRNN, self).__init__()

        self.num_classes = Config.NUM_CLASSES
        self.input_channels = Config.IN_CHANNELS

        # --- CNN Backbone ---
        # Input: (B, 1, 128, 2048)

        # Layer 1: Downsample H and W by 2
        # Input: 1 x 128 x 2048 -> Output: 64 x 64 x 1024
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Downsample H by 2, Keep W (Stride 2, 1)
        # Input: 64 x 64 x 1024 -> Output: 128 x 32 x 1024
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 3: Downsample H by 2, Keep W
        # Input: 128 x 32 x 1024 -> Output: 256 x 16 x 1024
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 4: Downsample H by 2, Keep W
        # Input: 256 x 16 x 1024 -> Output: 512 x 8 x 1024
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 5: Downsample H by 2, Keep W
        # Input: 512 x 8 x 1024 -> Output: 512 x 4 x 1024
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 6: Conv block without pooling
        # Input: 512 x 4 x 1024 -> Output: 512 x 4 x 1024
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # Final pooling to collapse Height to 1
        # Output: 512 x 1 x 1024
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, None))

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # --- RNN Encoder ---
        # Input features to RNN = CNN output channels (512)
        self.rnn_input_size = 512
        self.rnn_hidden_size = 256
        self.rnn_num_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

        # --- Transcription ---
        # Bidirectional output is 2 * hidden_size
        self.linear = nn.Linear(self.rnn_hidden_size * 2, self.num_classes)

    def forward(self, x):
        # x: (B, 1, 128, 2048)

        # CNN
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.adaptive_pool(x)
        # x: (B, 512, 1, 1024) -> Channels=512, Height=1, Width=1024 (Sequence Length)

        # Prepare for RNN
        # Remove height dimension: (B, 512, W)
        x = x.squeeze(2)

        # Permute to (B, W, C) for LSTM with batch_first=True
        x = x.permute(0, 2, 1)
        # x: (B, 1024, 512) -> (Batch, Seq_Len, Features)

        # RNN
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        # x: (B, 1024, 512) -> (Batch, Seq_Len, Hidden*2)

        # Linear Projection
        x = self.linear(x)
        # x: (B, 1024, Num_Classes)

        # Log Softmax for CTC Loss
        # CTC Loss expects log probabilities
        x = F.log_softmax(x, dim=2)

        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, targets, target_lengths) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        batch_size = images.size(0)

        # Forward pass
        # Output shape: (B, T, C)
        log_probs = model(images)

        # CTC Loss expects:
        # log_probs: (T, B, C) -> We need to permute
        # targets: (Sum of target lengths) or (B, max_target_len)
        # input_lengths: (B,) -> Length of each sequence in batch (constant here)
        # target_lengths: (B,) -> Length of each target sequence

        log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, C)

        # Input lengths are constant (T=1024) for all images in batch
        T = log_probs.size(1)
        input_lengths = torch.full(
            size=(batch_size,), fill_value=T, dtype=torch.long
        ).to(device)

        loss = criterion(log_probs_ctc, targets, input_lengths, target_lengths)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_NORM)
        optimizer.step()

        losses.update(loss.item(), batch_size)

    return losses.avg


def validate(model, loader, criterion, tokenizer, device):
    model.eval()
    losses = AverageMeter()
    levenshtein_scores = AverageMeter()

    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            targets_dev = targets.to(device)
            target_lengths = target_lengths.to(device)
            batch_size = images.size(0)

            # Forward
            log_probs = model(images)  # (B, T, C)
            log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, C)

            T = log_probs.size(1)
            input_lengths = torch.full(
                size=(batch_size,), fill_value=T, dtype=torch.long
            ).to(device)

            # Loss
            loss = criterion(log_probs_ctc, targets_dev, input_lengths, target_lengths)
            losses.update(loss.item(), batch_size)

            # Decoding for Metric
            # Get argmax indices: (B, T)
            preds = torch.argmax(log_probs, dim=2)
            decoded_preds = tokenizer.decode_batch(preds)

            # Decode targets (targets tensor is concatenated or padded, we need to split it)
            # The Tokenizer expects indices. Since targets might be a 1D concatenated tensor
            # (if using CTCLoss defaults) or 2D padded, we need to handle it.
            # ChemicalDataset returns 1D LongTensor for label_seq.
            # However, DataLoader collate_fn usually pads sequences if they are variable length.
            # Default collate pads with 0.

            # Reconstruct target strings
            decoded_targets = []
            targets_cpu = targets.cpu()
            # If targets is 2D (B, MaxLen)
            if targets_cpu.dim() == 2:
                for i in range(batch_size):
                    length = target_lengths[i].item()
                    seq = targets_cpu[i][:length]
                    # Convert to list of ints
                    seq_list = seq.tolist()
                    # Map to chars (skipping special tokens if any, but raw vocab is used)
                    # We use tokenizer.idx2char directly
                    t_str = "".join([tokenizer.idx2char[idx] for idx in seq_list])
                    decoded_targets.append(t_str)

            # Compute Metric
            score = compute_levenshtein(decoded_preds, decoded_targets)
            levenshtein_scores.update(score, batch_size)

    return losses.avg, levenshtein_scores.avg


def fit_model():
    print(f"Initializing Model on {Config.DEVICE}...")
    model = CRNN().to(Config.DEVICE)

    # Optimizer & Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # blank=0 as defined in Config
    criterion = nn.CTCLoss(blank=Config.BLANK_IDX, reduction="mean", zero_infinity=True)

    # Datasets
    print("Loading Datasets...")
    train_dataset = ChemicalDataset(mode="train", load_cached_data=True)
    val_dataset = ChemicalDataset(mode="val", load_cached_data=True)

    # Custom collate to pad sequences
    def collate_fn(batch):
        images, sequences, lengths = zip(*batch)
        images = torch.stack(images, 0)
        lengths = torch.stack(lengths, 0)
        # Pad sequences with 0 (blank_idx) or ignore_index?
        # CTCLoss handles padding if we provide lengths. We just need a tensor.
        # Pad with 0 (BLANK_IDX)
        padded_seqs = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=Config.BLANK_IDX
        )
        return images, padded_seqs, lengths

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    tokenizer = Tokenizer()

    best_lev = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_lev = validate(
            model, val_loader, criterion, tokenizer, Config.DEVICE
        )

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {duration:.0f}s | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Levenshtein: {val_lev:.5f}"
        )

        # Checkpoint
        if val_lev < best_lev:
            best_lev = val_lev
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Levenshtein: {best_lev:.5f})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def predict_and_submit(model_path):
    print("Loading Best Model for Inference...")
    model = CRNN().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    test_dataset = ChemicalDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    tokenizer = Tokenizer()
    results = []

    print("Generating Predictions...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(Config.DEVICE)

            # Forward
            log_probs = model(images)  # (B, T, C)

            # Greedy Decode
            preds = torch.argmax(log_probs, dim=2)  # (B, T)
            decoded_strs = tokenizer.decode_batch(preds)

            for img_id, pred_str in zip(image_ids, decoded_strs):
                results.append({"image_id": img_id, "InChI": pred_str})

    # Save Submission
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Total predictions: {len(df_sub)}")


def run_pipeline():
    best_model = fit_model()
    predict_and_submit(best_model)
