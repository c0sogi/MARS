import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import soundfile as sf
import torchaudio
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Audio Params
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4000 samples
    N_FFT = 1024
    HOP_LENGTH = 20  # 10ms at 2000Hz
    N_MELS = 384
    FMIN = 0
    FMAX = None

    # Model Params
    BACKBONE = "tf_efficientnetv2_m"
    PRETRAINED = True
    NUM_CLASSES = 1

    # Training Params
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 20  # Per stage
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # Early stopping
    POS_WEIGHT = 9.0  # Inverse class frequency approx

    # Augmentation
    MIXUP_ALPHA = 0.4
    SPECAUG_TIME_MASK = 20
    SPECAUG_FREQ_MASK = 20

    # Checkpoints
    TEACHER_WEIGHTS = os.path.join(WORKING_DIR, "teacher_best.pth")
    STUDENT_WEIGHTS = os.path.join(WORKING_DIR, "student_best.pth")

    @classmethod
    def setup(cls):
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ==========================================
# DATA PROCESSING
# ==========================================


def load_audio(file_path):
    """Loads audio and pads/crops to fixed length."""
    try:
        wav, sr = sf.read(file_path)
        # Ensure fixed length
        if len(wav) < Config.N_SAMPLES:
            pad_width = Config.N_SAMPLES - len(wav)
            wav = np.pad(wav, (0, pad_width), mode="constant")
        else:
            wav = wav[: Config.N_SAMPLES]
        return wav.astype(np.float32)
    except Exception as e:
        return np.zeros(Config.N_SAMPLES, dtype=np.float32)


def compute_melspec(waveform):
    """Computes Log-Mel Spectrogram with Instance Normalization."""
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        power=2.0,
    )

    spec = mel_transform(waveform_tensor)
    spec = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(spec)
    spec = spec.squeeze(0).numpy()

    # Instance Min-Max Normalization
    min_val = spec.min()
    max_val = spec.max()
    if max_val - min_val > 1e-6:
        spec = (spec - min_val) / (max_val - min_val)
    else:
        spec = np.zeros_like(spec)

    return spec


def process_data_to_cache(metadata_path, cache_name, load_cached=True):
    """Loads metadata, reads audio, computes specs, and caches to disk."""
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npz")

    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["specs"], data["labels"], data["clips"]

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    specs = []
    labels = []
    clips = []

    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        wav = load_audio(full_path)
        spec = compute_melspec(wav)

        specs.append(spec)
        clips.append(row["clip_name"])

        if "label" in row:
            labels.append(row["label"])
        else:
            labels.append(-1)

    specs = np.stack(specs)
    labels = np.array(labels)
    clips = np.array(clips)

    np.savez(cache_path, specs=specs, labels=labels, clips=clips)
    print(f"Saved processed data to {cache_path}")
    return specs, labels, clips


class WhaleDataset(Dataset):
    def __init__(
        self, specs, labels, transform=None, is_test=False, pseudo_labels=None
    ):
        self.specs = specs
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]

        # Convert to tensor and expand to 3 channels
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        spec = spec.repeat(3, 1, 1)

        # Apply transforms (SpecAugment)
        if self.transform:
            spec = self.transform(spec)

        if self.is_test:
            return spec, self.labels[idx]

        # Determine target
        if self.pseudo_labels is not None:
            target = self.pseudo_labels[idx]
        else:
            target = self.labels[idx]

        return spec, torch.tensor(target, dtype=torch.float32)


# ==========================================
# MODEL
# ==========================================


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class WhaleModel(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super(WhaleModel, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine in_features dynamically
        dummy = torch.randn(
            1, 3, Config.N_MELS, Config.N_SAMPLES // Config.HOP_LENGTH + 1
        )
        with torch.no_grad():
            feats = self.backbone(dummy)
            in_features = feats.shape[1]

        self.pooling = GeM()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ==========================================
# TRAINING ENGINE
# ==========================================


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, Config.MIXUP_ALPHA
            )
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc, all_preds


def run_training_stage(
    train_specs,
    train_labels,
    val_specs,
    val_labels,
    save_path,
    pseudo_labels=None,
    device="cuda",
):

    # Setup Datasets
    train_dataset = WhaleDataset(train_specs, train_labels, pseudo_labels=pseudo_labels)
    val_dataset = WhaleDataset(val_specs, val_labels)

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

    # Setup Model & Opt
    model = WhaleModel().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(Config.POS_WEIGHT).to(device)
    )
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training -> {save_path}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_mixup=True
        )
        val_loss, val_auc, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(save_path))
    return model, best_auc


def generate_predictions(model, specs, device="cuda"):
    dataset = WhaleDataset(specs, np.zeros(len(specs)), is_test=True)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_pipeline(load_cached=True):
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    train_specs, train_labels, _ = process_data_to_cache(
        Config.TRAIN_METADATA, "train", load_cached
    )
    val_specs, val_labels, _ = process_data_to_cache(
        Config.VAL_METADATA, "val", load_cached
    )
    test_specs, _, test_clips = process_data_to_cache(
        Config.TEST_METADATA, "test", load_cached
    )

    # 2. Teacher Training
    print("\n=== Stage 1: Teacher Training ===")
    teacher_model, teacher_auc = run_training_stage(
        train_specs,
        train_labels,
        val_specs,
        val_labels,
        Config.TEACHER_WEIGHTS,
        device=device,
    )
    print(f"Teacher Best AUC: {teacher_auc:.10f}")

    # 3. Pseudo-Labeling
    print("\n=== Stage 2: Pseudo-Labeling ===")
    test_probs = generate_predictions(teacher_model, test_specs, device=device)

    # Combine Data: Hard labels for train, Soft labels for test
    combined_specs = np.concatenate([train_specs, test_specs])
    combined_labels = np.concatenate(
        [train_labels, np.zeros(len(test_specs))]
    )  # Dummy labels

    train_pseudo = train_labels.astype(np.float32)
    test_pseudo = test_probs.astype(np.float32)
    combined_pseudo = np.concatenate([train_pseudo, test_pseudo])

    # 4. Student Training
    print("\n=== Stage 3: Student Training ===")
    student_model, student_auc = run_training_stage(
        combined_specs,
        combined_labels,
        val_specs,
        val_labels,
        Config.STUDENT_WEIGHTS,
        pseudo_labels=combined_pseudo,
        device=device,
    )
    print(f"Student Best AUC: {student_auc:.10f}")

    # 5. Submission
    print("\n=== Generating Submission ===")
    final_probs = generate_predictions(student_model, test_specs, device=device)

    df_sub = pd.DataFrame({"clip": test_clips, "probability": final_probs})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
