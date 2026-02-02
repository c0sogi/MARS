import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.dataset import CervicalSpineDataset

# ==========================================
# Model Architecture
# ==========================================


class GatedContextBlock(nn.Module):
    """
    Applies a Gated Linear Unit (GLU) convolution to the sequence of slice features.
    Input: (Batch, Seq_Len, Features)
    Output: (Batch, Seq_Len, Features)
    """

    def __init__(self, in_channels):
        super().__init__()
        # Conv1d expects (Batch, Channels, Length)
        # We use kernel_size=3, padding=1 to maintain sequence length
        self.content_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, padding=1
        )
        self.gate_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, S, C) -> Permute to (B, C, S)
        x_perm = x.permute(0, 2, 1)

        content = self.content_conv(x_perm)
        gate = self.sigmoid(self.gate_conv(x_perm))

        # Gating mechanism: Element-wise multiplication
        out = content * gate

        # Normalization
        out = self.bn(out)

        # Permute back to (B, S, C)
        return out.permute(0, 2, 1)


class CervicalSpineMIL(nn.Module):
    """
    2.5D Multiple Instance Learning Network with Gated Context.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Backbone: ResNet18
        # We use the default weights (ImageNet)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove the fully connected layer (fc)
        # The output of the layer before fc (avgpool) is (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        # 2. Context Module
        self.context = GatedContextBlock(self.feature_dim)

        # 3. Instance Classifier
        # Predicts 7 logits (C1-C7) per slice
        self.classifier = nn.Linear(self.feature_dim, 7)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, 3, H, W)
        Returns:
            dict: containing 'vertebrae_logits' (B, 7) and 'patient_logit' (B, 1)
        """
        b, s, c, h, w = x.shape

        # Fold Batch and Seq_Len dimensions to process slices in parallel
        x_reshaped = x.view(b * s, c, h, w)

        # Extract features
        # Backbone output: (B*S, 512, 1, 1) -> Flatten to (B*S, 512)
        features = self.backbone(x_reshaped)
        features = features.view(b * s, -1)

        # Unfold back to sequence: (B, S, 512)
        features = features.view(b, s, self.feature_dim)

        # Apply Gated Context (mixes information along Z-axis)
        features = self.context(features)

        # Instance-level classification: (B, S, 7)
        instance_logits = self.classifier(features)

        # Aggregation: Global Max Pooling over the sequence
        # study_logits: (B, 7)
        study_logits, _ = torch.max(instance_logits, dim=1)

        # Patient Overall Prediction
        # Derived as the maximum of the 7 vertebral logits
        # This enforces consistency: if C_i is high, patient_overall must be high.
        patient_logit, _ = torch.max(study_logits, dim=1, keepdim=True)

        return {"vertebrae_logits": study_logits, "patient_logit": patient_logit}


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion_bce, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        labels_vert = batch["labels"]["vertebrae"].to(device)
        labels_patient = batch["labels"]["patient_overall"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        pred_vert = outputs["vertebrae_logits"]
        pred_patient = outputs["patient_logit"]

        # Hierarchical Compound Loss
        # 1. Average loss across 7 vertebrae
        loss_vert = criterion_bce(pred_vert, labels_vert)
        # 2. Patient overall loss
        loss_patient = criterion_bce(pred_patient, labels_patient)

        # Implicit weighting: Mean(Vert) + Patient
        # This gives patient_overall roughly 50% of the weight
        loss = loss_vert + loss_patient

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion_bce, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels_vert = batch["labels"]["vertebrae"].to(device)
            labels_patient = batch["labels"]["patient_overall"].to(device).unsqueeze(1)

            outputs = model(images)
            pred_vert = outputs["vertebrae_logits"]
            pred_patient = outputs["patient_logit"]

            loss_vert = criterion_bce(pred_vert, labels_vert)
            loss_patient = criterion_bce(pred_patient, labels_patient)
            loss = loss_vert + loss_patient

            total_loss += loss.item()

    return total_loss / len(loader)


def run_training():
    print("Starting Training Pipeline...")
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)

    # 1. Datasets & Loaders
    train_dataset = CervicalSpineDataset(
        Config.TRAIN_METADATA_PATH, phase="train", load_cached_data=True
    )
    val_dataset = CervicalSpineDataset(
        Config.VAL_METADATA_PATH, phase="val", load_cached_data=True
    )

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

    # 2. Model setup
    model = CervicalSpineMIL(pretrained=Config.PRETRAINED).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Decoupled Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.NUM_EPOCHS * Config.T_MAX_MULTIPLIER)
    )

    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    best_val_loss = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with Val Loss: {val_loss:.6f}")

    print("Training completed.")


def run_inference():
    print("Starting Inference Pipeline...")
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # Test metadata contains unique studies
    test_dataset = CervicalSpineDataset(
        Config.TEST_METADATA_PATH, phase="test", load_cached_data=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Model
    model = CervicalSpineMIL(pretrained=False).to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No trained model found. Using random weights.")

    model.eval()

    # 3. Generate Predictions
    study_predictions = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inferring"):
            images = batch["image"].to(device)
            study_ids = batch["study_id"]

            outputs = model(images)

            # Apply Sigmoid to get probabilities
            probs_vert = torch.sigmoid(outputs["vertebrae_logits"]).cpu().numpy()
            probs_patient = torch.sigmoid(outputs["patient_logit"]).cpu().numpy()

            for i, study_id in enumerate(study_ids):
                preds = {
                    "C1": probs_vert[i, 0],
                    "C2": probs_vert[i, 1],
                    "C3": probs_vert[i, 2],
                    "C4": probs_vert[i, 3],
                    "C5": probs_vert[i, 4],
                    "C6": probs_vert[i, 5],
                    "C7": probs_vert[i, 6],
                    "patient_overall": probs_patient[i, 0],
                }
                study_predictions[study_id] = preds

    # 4. Format Submission
    # Load the sample submission or test.csv to get the required row_ids
    # input/test.csv contains the mapping: row_id -> StudyInstanceUID, prediction_type
    test_df_path = os.path.join(Config.INPUT_DIR, "test.csv")
    if os.path.exists(test_df_path):
        test_df = pd.read_csv(test_df_path)
    else:
        # Fallback to sample submission if test.csv is missing (unlikely)
        test_df = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))
        # Parse row_id to get study_id and prediction_type
        # Format: study_id + "_" + prediction_type
        # This is a bit risky if underscores exist in UID, but standard DICOM UIDs use dots
        # We assume test.csv is available as per problem description.

    # We need to fill the 'fractured' column
    results = []

    # If test.csv has 'prediction_type' column (as per description)
    if "prediction_type" in test_df.columns:
        for idx, row in test_df.iterrows():
            study_id = row["StudyInstanceUID"]
            pred_type = row["prediction_type"]
            row_id = row["row_id"]

            # Retrieve prediction
            if study_id in study_predictions:
                prob = study_predictions[study_id].get(pred_type, 0.5)
            else:
                prob = 0.5  # Fallback

            results.append({"row_id": row_id, "fractured": prob})
    else:
        # Fallback parsing from row_id
        for idx, row in test_df.iterrows():
            row_id = row["row_id"]
            # Split from the right to handle prediction types with underscores (patient_overall)
            parts = row_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1] == "overall":
                # Handle 'patient_overall' which splits into ['..._patient', 'overall']
                # Actually split by study ID logic is safer if we know the UID format
                # But usually it is {StudyID}_{Type}
                # Let's try to match known types
                for k in ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
                    if row_id.endswith(f"_{k}"):
                        study_id = row_id.replace(f"_{k}", "")
                        if study_id in study_predictions:
                            prob = study_predictions[study_id][k]
                            results.append({"row_id": row_id, "fractured": prob})
                        break

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Execute Training
    run_training()

    # Execute Inference
    run_inference()


if __name__ == "__main__":
    main()
