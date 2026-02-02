import os
import glob
import random
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import LabelEncoder


class Config:
    # Reproducibility
    SEED = 42

    # Audio Parameters
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160

    # Training Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MIXUP_ALPHA = 1.0

    # Model / Optimization
    EMA_DECAY = 0.999
    DROPOUT_RATE = 0.5
    NUM_DROPOUTS = 8  # Multi-Sample Dropout

    # Paths
    INPUT_ROOT = "./input"
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"
    CACHE_DIR = "./working/idea_16"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Labels
    TARGET_LABELS = {
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    }
    SILENCE_LABEL = "silence"
    UNKNOWN_LABEL = "unknown"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------------------------------------------------------
# Data Processing & Dataset
# -----------------------------------------------------------------------------


def get_fine_grained_labels(df, input_root):
    """
    Extracts fine-grained labels from filepaths.
    The provided metadata has 'label' column with simplified classes (target + unknown).
    We need the original folder names for the 31+ class auxiliary task.
    """
    fine_labels = []
    for idx, row in df.iterrows():
        if row["label"] == Config.SILENCE_LABEL:
            fine_labels.append(Config.SILENCE_LABEL)
        else:
            # filepath format: train/audio/<label>/<file>.wav
            # We extract the parent directory name
            rel_path = row["filepath"]
            parts = rel_path.split(os.sep)
            # Assuming structure train/audio/label/file.wav
            # parts[-2] should be the label
            label = parts[-2]
            fine_labels.append(label)
    return fine_labels


def load_or_create_metadata(load_cached_data=True):
    """
    Loads metadata and augments it with fine-grained labels.
    Uses caching as requested.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, "metadata_fine.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached metadata from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Processing metadata from scratch...")
    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Extract fine-grained labels
    df_train["fine_label"] = get_fine_grained_labels(df_train, Config.INPUT_ROOT)
    df_val["fine_label"] = get_fine_grained_labels(df_val, Config.INPUT_ROOT)

    # Combine for encoding consistency
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    # Save to cache
    df_all.to_parquet(cache_path)
    return df_all


class SpeechDataset(Dataset):
    def __init__(
        self,
        df,
        label_encoder=None,
        transform=None,
        is_train=True,
        background_noise_paths=None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform
        self.is_train = is_train
        self.background_noise_paths = background_noise_paths

        # Audio settings
        self.sr = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.target_len = int(self.sr * self.duration)

        # Mel Spectrogram Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def load_audio(self, filepath):
        # Handle silence generation from background noise
        if "silence" in filepath or "background_noise" in filepath:
            # If it's a specific noise file (training), load it.
            # If it's a placeholder (inference), we might generate silence.
            # Here we assume filepath is valid relative path.
            pass

        full_path = os.path.join(Config.INPUT_ROOT, filepath)

        # If label is silence and path is generic or noise, we might need special handling
        # But metadata points to actual files in _background_noise_ for silence samples in train?
        # The provided metadata script maps _background_noise_ files to 'silence'.
        # However, _background_noise_ files are long. We need to crop them.

        try:
            wav, sr = torchaudio.load(full_path)
        except Exception:
            # Fallback for missing files (should not happen given checks)
            return torch.zeros(1, self.target_len)

        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav)

        return wav

    def pad_or_crop(self, wav):
        # wav shape: (channels, time)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        length = wav.shape[1]

        if length < self.target_len:
            padding = self.target_len - length
            # Random padding for train, center for val
            if self.is_train:
                offset = random.randint(0, padding)
            else:
                offset = padding // 2
            wav = F.pad(wav, (offset, padding - offset))
        elif length > self.target_len:
            # Random crop for train, center for val
            if self.is_train:
                offset = random.randint(0, length - self.target_len)
            else:
                offset = (length - self.target_len) // 2
            wav = wav[:, offset : offset + self.target_len]

        return wav

    def add_noise(self, wav):
        # Simple noise injection
        if self.background_noise_paths and random.random() < 0.5:
            noise_path = random.choice(self.background_noise_paths)
            noise_wav = self.load_audio(noise_path)

            # Ensure noise is long enough
            while noise_wav.shape[1] < self.target_len:
                noise_wav = torch.cat([noise_wav, noise_wav], dim=1)

            # Random crop noise
            start = random.randint(0, noise_wav.shape[1] - self.target_len)
            noise_crop = noise_wav[:, start : start + self.target_len]

            snr_db = random.uniform(10, 30)
            signal_rms = wav.pow(2).mean().sqrt()
            noise_rms = noise_crop.pow(2).mean().sqrt()

            if noise_rms > 0:
                scale = signal_rms / (noise_rms * (10 ** (snr_db / 20)))
                wav = wav + scale * noise_crop

        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["fine_label"] if "fine_label" in row else "unknown"

        wav = self.load_audio(filepath)
        wav = self.pad_or_crop(wav)

        if self.is_train:
            wav = self.add_noise(wav)

        # To Spectrogram
        spec = self.mel_transform(wav)
        spec = self.db_transform(spec)

        # SpecAugment (Time/Freq Masking)
        if self.is_train:
            # Freq Masking
            if random.random() < 0.5:
                spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)(spec)
            # Time Masking
            if random.random() < 0.5:
                spec = torchaudio.transforms.TimeMasking(time_mask_param=35)(spec)

        # Encode Label
        label = 0
        if self.label_encoder:
            label = self.label_encoder.transform([label_str])[0]

        return spec, label


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class AttentivePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, T)
        # Attention weights
        w = self.attn(x)  # (B, 1, T)
        w = F.softmax(w, dim=2)
        # Weighted sum
        x = torch.sum(x * w, dim=2)  # (B, C)
        return x


class DilatedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load EfficientNet B2
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=True, features_only=False
        )

        # 1. Modify Input Layer (3 channels -> 1 channel)
        original_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Average weights: (Out, 3, K, K) -> (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(original_conv.weight, dim=1, keepdim=True)
        self.backbone.conv_stem = new_conv

        # 2. Dilated Convolutions in the last stage
        # EfficientNet-B2 blocks structure. We iterate and modify the last blocks.
        # This is heuristic based on timm structure.
        # Usually blocks are in self.backbone.blocks
        # We target the last stage blocks.
        last_stage_idx = len(self.backbone.blocks) - 1
        for block in self.backbone.blocks[last_stage_idx]:
            # Modify conv_dw or conv_pw depending on block type (usually MBConv)
            # We set dilation=2 and stride=1 if stride was 2
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    # Heuristic: if it's the depthwise conv
                    if m.groups == m.in_channels and m.in_channels > 1:
                        m.dilation = (2, 2)
                        m.padding = (2, 2)  # Adjust padding for dilation 2, kernel 3
                        # If stride was 2, set to 1 to preserve resolution
                        if m.stride == (2, 2):
                            m.stride = (1, 1)

        # 3. Pooling & Head
        self.pool = AttentivePooling(self.backbone.num_features)

        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.DROPOUT_RATE) for _ in range(Config.NUM_DROPOUTS)]
        )
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.backbone.forward_features(x)  # (B, C, F', T')

        # Collapse Frequency dimension (Global Avg Pool over Freq) or Flatten
        # EfficientNet features are 4D. We want (B, C, T) for attentive pooling over time.
        # Usually we pool over frequency first.
        x = torch.mean(x, dim=2)  # (B, C, T')

        x = self.pool(x)  # (B, C)

        # Multi-Sample Dropout
        logits = []
        for dropout in self.dropouts:
            logits.append(self.fc(dropout(x)))

        # Average logits
        return torch.mean(torch.stack(logits), dim=0)


# -----------------------------------------------------------------------------
# Utilities: EMA, Mixup
# -----------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.ema.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -----------------------------------------------------------------------------
# Training & Inference
# -----------------------------------------------------------------------------


def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Preparation
    df_all = load_or_create_metadata(load_cached_data=True)
    df_train = df_all[df_all["split"] == "train"].copy()
    df_val = df_all[df_all["split"] == "val"].copy()

    # Encode Labels
    le = LabelEncoder()
    le.fit(df_train["fine_label"])
    num_classes = len(le.classes_)
    print(f"Training on {num_classes} fine-grained classes.")

    # Background noise paths for injection
    bg_noise_dir = os.path.join(
        Config.INPUT_ROOT, "train", "audio", "_background_noise_"
    )
    bg_files = (
        glob.glob(os.path.join(bg_noise_dir, "*.wav"))
        if os.path.exists(bg_noise_dir)
        else []
    )

    # Weighted Sampler
    # Target labels get higher weight
    class_counts = df_train["fine_label"].value_counts()
    weights = []
    for label in df_train["fine_label"]:
        count = class_counts[label]
        if label in Config.TARGET_LABELS:
            # Upsample target to ~2000 if it were uniform, but here we just weight inversely
            # We want targets to be frequent.
            # Simple strategy: Weight = 1/count * Boost
            # But the Idea says "Target commands upsampled to ~2000".
            # If count is ~1700, weight ~ 1.2.
            # If auxiliary is ~100, weight ~ 1.0 (natural).
            # Let's just use inverse freq with a boost for targets.
            w = 1.0 / count
        else:
            w = 1.0 / count
        weights.append(w)

    sampler = WeightedRandomSampler(
        weights, num_samples=len(df_train), replacement=True
    )

    train_ds = SpeechDataset(
        df_train, le, is_train=True, background_noise_paths=bg_files
    )
    val_ds = SpeechDataset(df_val, le, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Model Setup
    model = DilatedEfficientNet(num_classes).to(device)
    ema = ModelEMA(model, Config.EMA_DECAY)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA
            )

            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            loss.backward()
            optimizer.step()
            ema.update(model)

            train_loss += loss.item()

        scheduler.step()

        # Validation (using EMA)
        ema.ema.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = ema.ema(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {acc:.6f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                ema.ema.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )

    print(f"Best Validation Accuracy: {best_acc:.6f}")
    return le


def generate_submission():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata / encoder
    df_all = load_or_create_metadata(load_cached_data=True)
    le = LabelEncoder()
    le.fit(df_all[df_all["split"].isin(["train", "val"])]["fine_label"])
    num_classes = len(le.classes_)

    # Load Test Data
    df_test = pd.read_csv(Config.TEST_METADATA)
    # Placeholder fine_label for dataset compatibility
    df_test["fine_label"] = "unknown"

    test_ds = SpeechDataset(df_test, le, is_train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Model
    model = DilatedEfficientNet(num_classes).to(device)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("No trained model found. Cannot generate submission.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    fnames = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

            predicted_labels = le.inverse_transform(predicted_indices.cpu().numpy())
            predictions.extend(predicted_labels)

    # Post-processing mapping
    final_labels = []
    for label in predictions:
        if label in Config.TARGET_LABELS:
            final_labels.append(label)
        elif label == Config.SILENCE_LABEL:
            final_labels.append(Config.SILENCE_LABEL)
        else:
            final_labels.append(Config.UNKNOWN_LABEL)

    # Save
    submission = pd.DataFrame(
        {"fname": df_test["filepath"].apply(os.path.basename), "label": final_labels}
    )

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
