import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import DataLoader

from library.config import (
    INPUT_DIR,
    TEST_METADATA_PATH,
    LABEL2ID,
    ID2LABEL,
    UNKNOWN_LABEL,
    SILENCE_LABEL,
    TARGET_LABELS,
    BATCH_SIZE,
    NUM_WORKERS,
    CONFIDENCE_THRESHOLD,
    WORKING_DIR,
    SAMPLE_RATE,
    map_prediction_to_submission,
)
from library.dataset import SpeechDataset
from library.utils import Mixup


class PseudoLabelDataset(SpeechDataset):
    """
    Dataset for Pseudo-Labeled data.
    Overrides __getitem__ to read labels from the DataFrame directly
    instead of inferring from directory structure.
    """

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(INPUT_DIR, row["filepath"])

        # 1. Load Audio (reuse parent method)
        waveform = self._load_audio(filepath)

        # 2. Determine Label (Directly from DF)
        label_str = row["label"]
        label_id = LABEL2ID.get(label_str, LABEL2ID[UNKNOWN_LABEL])

        # 3. Noise Injection
        # We inject noise if it's training mode and not silence
        if self.mode == "train" and self.noise_files and label_str != SILENCE_LABEL:
            if torch.rand(1).item() < 0.8:
                waveform = self._inject_noise(waveform)

        # 4. Generate Spectrogram
        spec = self.mel_transform(waveform)
        spec = self.amp_to_db(spec)

        # 5. SpecAugment
        if self.mode == "train":
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)

        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        return spec, torch.tensor(label_id, dtype=torch.long)


def load_noise_files():
    """
    Loads background noise files for augmentation.
    """
    noise_dir = os.path.join(INPUT_DIR, "train", "audio", "_background_noise_")
    noise_files = []
    if os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.endswith(".wav"):
                try:
                    w, s = torchaudio.load(os.path.join(noise_dir, f))
                    if s != SAMPLE_RATE:
                        w = torchaudio.transforms.Resample(s, SAMPLE_RATE)(w)
                    if w.shape[0] > 1:
                        w = torch.mean(w, dim=0, keepdim=True)
                    noise_files.append(w)
                except:
                    continue
    return noise_files


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_fn):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_fn.criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    avg_loss = running_loss / dataset_size
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Accuracy: {accuracy}")

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    mixup_fn,
):
    """
    Full training loop with Early Stopping.
    """
    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup_fn
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Acc: {val_acc}"
        )

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return model


def generate_pseudo_labels(model, device, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Generates pseudo-labels for the test set.
    Returns a DataFrame with columns ['filepath', 'label', 'score', 'subject_id'].
    """
    model.eval()

    # Load Test Metadata
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # Create Test Dataset (Standard SpeechDataset works for inference)
    test_ds = SpeechDataset(df_test, mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    pseudo_records = []

    with torch.no_grad():
        start_idx = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            max_probs, preds = torch.max(probs, dim=1)

            max_probs = max_probs.cpu().numpy()
            preds = preds.cpu().numpy()

            for i in range(batch_size):
                score = max_probs[i]
                if score >= confidence_threshold:
                    pred_id = preds[i]
                    label_str = ID2LABEL[pred_id]

                    global_idx = start_idx + i
                    filepath = df_test.iloc[global_idx]["filepath"]

                    pseudo_records.append(
                        {
                            "filepath": filepath,
                            "label": label_str,
                            "score": score,
                            "subject_id": "pseudo_gen",
                        }
                    )

            start_idx += batch_size

    df_pseudo = pd.DataFrame(pseudo_records)
    return df_pseudo


def generate_submission(model, device, output_path="./submission/submission.csv"):
    """
    Generates the final submission file.
    """
    model.eval()
    df_test = pd.read_csv(TEST_METADATA_PATH)
    test_ds = SpeechDataset(df_test, mode="test")
    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    with torch.no_grad():
        start_idx = 0
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()

            for i in range(len(preds)):
                pred_id = preds[i]
                label_str = ID2LABEL[pred_id]

                final_label = map_prediction_to_submission(label_str)

                fname = os.path.basename(df_test.iloc[start_idx + i]["filepath"])
                results.append({"fname": fname, "label": final_label})

            start_idx += len(preds)

    df_sub = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
