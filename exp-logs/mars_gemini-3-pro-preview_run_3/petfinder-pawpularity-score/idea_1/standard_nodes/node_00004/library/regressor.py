import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from library.config import Config
from library.utils import compute_rmse, save_submission, set_seed


class PawpularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Remove the classifier to get features
        self.backbone.classifier = nn.Identity()

        # Feature dimension of MobileNetV2 is 1280
        # Metadata dimension is 12
        self.fc = nn.Linear(1280 + 12, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, image, meta):
        # Extract image features
        x = self.backbone(image)  # (Batch, 1280)
        x = self.dropout(x)

        # Concatenate with metadata
        combined = torch.cat([x, meta], dim=1)

        # Predict
        out = self.fc(combined)
        return out.squeeze()


class FineTuningRegressor:
    """
    A wrapper for end-to-end fine-tuning of the MobileNetV2 model.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = PawpularityModel().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def fit(self, train_loader, val_loader):
        print(f"Starting fine-tuning for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0.0

            for images, meta, targets, _ in train_loader:
                images, meta, targets = (
                    images.to(self.device),
                    meta.to(self.device),
                    targets.to(self.device),
                )

                self.optimizer.zero_grad()
                outputs = self.model(images, meta)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)
            train_rmse = np.sqrt(avg_train_loss)

            # Validation
            val_rmse = self.evaluate(val_loader)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}"
            )

    def evaluate(self, loader):
        self.model.eval()
        mse_sum = 0.0
        with torch.no_grad():
            for images, meta, targets, _ in loader:
                images, meta, targets = (
                    images.to(self.device),
                    meta.to(self.device),
                    targets.to(self.device),
                )
                outputs = self.model(images, meta)
                loss = self.criterion(outputs, targets)
                mse_sum += loss.item() * images.size(0)

        return np.sqrt(mse_sum / len(loader.dataset))

    def predict(self, loader):
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for images, meta, _, _ in loader:
                images, meta = images.to(self.device), meta.to(self.device)
                outputs = self.model(images, meta)
                preds_list.append(outputs.cpu().numpy())

        preds = np.concatenate(preds_list)
        return np.clip(preds, 1.0, 100.0)
