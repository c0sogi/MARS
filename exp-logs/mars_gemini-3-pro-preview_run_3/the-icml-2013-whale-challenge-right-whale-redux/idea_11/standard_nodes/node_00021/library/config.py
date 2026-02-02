import os
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import timm
from sklearn.metrics import roc_auc_score
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore")


class Config:
    """
    Central configuration for the Right Whale Detection pipeline.
    """

    # File Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Audio Parameters
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds
    TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

    # Stream 1: Spectral (High Frequency Resolution)
    # Focus: Pitch contours, harmonic structure
    # Window: 50ms (100 samples), Hop: 25ms (50 samples)
    # n_fft=1024 allows for 513 freq bins, supporting 384 Mel bands
    S1_N_MELS = 384
    S1_N_FFT = 1024
    S1_WIN_LEN = 100
    S1_HOP_LEN = 50

    # Stream 2: Temporal (High Temporal Resolution)
    # Focus: Transient attacks, rapid dynamics
    # Window: 10ms (20 samples), Hop: 5ms (10 samples)
    # n_fft=256 allows for 129 freq bins, supporting 128 Mel bands
    S2_N_MELS = 128
    S2_N_FFT = 256
    S2_WIN_LEN = 20
    S2_HOP_LEN = 10

    # Model Hyperparameters
    BACKBONE = "tf_efficientnetv2_m"
    PRETRAINED = True
    DROPOUT = 0.3
    GEM_P = 3.0

    # Training Settings
    SEED = 42
    BATCH_SIZE = 16  # Reduced from 64 to prevent OOM
    EPOCHS = 25
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    MIXUP_ALPHA = 0.4
    POS_WEIGHT = 9.0  # Inverse class frequency
    PATIENCE = 5  # Early stopping patience

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets random seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Data Processing & Caching
# -----------------------------------------------------------------------------


def compute_spectrograms(waveform, config):
    """
    Computes both High-Frequency and High-Temporal resolution spectrograms.
    Returns normalized tensors.
    """
    # Ensure correct length
    if waveform.shape[1] < config.TARGET_LENGTH:
        pad = config.TARGET_LENGTH - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, : config.TARGET_LENGTH]

    # Stream 1: Spectral
    spec1_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.S1_N_FFT,
        win_length=config.S1_WIN_LEN,
        hop_length=config.S1_HOP_LEN,
        n_mels=config.S1_N_MELS,
        power=2.0,
    )
    spec1 = spec1_transform(waveform)
    spec1 = torchaudio.transforms.AmplitudeToDB()(spec1)

    # Stream 2: Temporal
    spec2_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.S2_N_FFT,
        win_length=config.S2_WIN_LEN,
        hop_length=config.S2_HOP_LEN,
        n_mels=config.S2_N_MELS,
        power=2.0,
    )
    spec2 = spec2_transform(waveform)
    spec2 = torchaudio.transforms.AmplitudeToDB()(spec2)

    # Instance-level Min-Max Normalization
    def normalize(s):
        min_val = s.min()
        max_val = s.max()
        if max_val - min_val > 1e-6:
            return (s - min_val) / (max_val - min_val)
        return torch.zeros_like(s)

    return normalize(spec1), normalize(spec2)


def prepare_data(metadata_df, config=Config, cache_name="train", load_cached=True):
    """
    Loads audio, computes dual spectrograms, and caches the result to disk.
    If load_cached is True and file exists, loads from disk.
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{cache_name}.npz")

    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path)
        return data["x1"], data["x2"], data["y"], data["clips"]

    print(f"Processing {len(metadata_df)} files for {cache_name}...")
    x1_list, x2_list, y_list, clips_list = [], [], [], []

    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)

            # Compute features
            s1, s2 = compute_spectrograms(waveform, config)

            x1_list.append(s1.numpy())
            x2_list.append(s2.numpy())

            # Handle labels (test set might not have them)
            if "label" in row:
                y_list.append(row["label"])
            else:
                y_list.append(-1)  # Dummy for test

            clips_list.append(row["clip_name"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Convert to numpy arrays
    # Shape: (N, 1, F, T)
    X1 = np.concatenate([x[np.newaxis, ...] for x in x1_list], axis=0)
    X2 = np.concatenate([x[np.newaxis, ...] for x in x2_list], axis=0)
    Y = np.array(y_list, dtype=np.float32)
    Clips = np.array(clips_list)

    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(cache_path, x1=X1, x2=X2, y=Y, clips=Clips)

    return X1, X2, Y, Clips


class WhaleDataset(Dataset):
    def __init__(self, x1, x2, y, augment=False):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.augment = augment

        # Augmentations
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=40)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec1 = torch.from_numpy(self.x1[idx])
        spec2 = torch.from_numpy(self.x2[idx])
        label = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.augment:
            # Apply SpecAugment to both streams independently
            spec1 = self.freq_mask(spec1)
            spec1 = self.time_mask(spec1)

            spec2 = self.freq_mask(spec2)
            spec2 = self.time_mask(spec2)

        return spec1, spec2, label


# -----------------------------------------------------------------------------
# Training Utilities
# -----------------------------------------------------------------------------


def mixup_data(x1, x2, y, alpha=0.4, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.unsqueeze(1)) + (1 - lam) * criterion(
        pred, y_b.unsqueeze(1)
    )


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0

    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Apply Mixup
        x1, x2, y_a, y_b, lam = mixup_data(x1, x2, y, config.MIXUP_ALPHA, device)

        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x1.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            outputs = model(x1, x2)
            loss = criterion(outputs, y.unsqueeze(1))

            running_loss += loss.item() * x1.size(0)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def inference(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x1, x2, _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            outputs = model(x1, x2)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds).flatten()


def run_training(config=Config):
    """
    Main training loop implementation.
    """
    Config.setup()
    device = torch.device(config.DEVICE)

    # Load Metadata
    train_df = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))

    # Prepare Data
    print("Preparing Training Data...")
    X1_train, X2_train, Y_train, _ = prepare_data(train_df, config, "train")
    print("Preparing Validation Data...")
    X1_val, X2_val, Y_val, _ = prepare_data(val_df, config, "val")

    # Datasets & Loaders
    train_dataset = WhaleDataset(X1_train, X2_train, Y_train, augment=True)
    val_dataset = WhaleDataset(X1_val, X2_val, Y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = DualStreamEfficientNet(config).to(device)

    # Weighted BCE Loss
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f} | "
            f"Time: {time.time() - start_time:.1f}s"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> New Best AUC! Model saved.")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc:.6f}")
    return best_model_path


def generate_submission(model_path, config=Config):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = torch.device(config.DEVICE)
    test_df = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

    print("Preparing Test Data...")
    X1_test, X2_test, Y_test, clips = prepare_data(test_df, config, "test")

    test_dataset = WhaleDataset(X1_test, X2_test, Y_test, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = DualStreamEfficientNet(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Inference
    print("Running Inference...")
    probs = inference(model, test_loader, device)

    # Save Submission
    submission = pd.DataFrame({"clip": clips, "probability": probs})

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
