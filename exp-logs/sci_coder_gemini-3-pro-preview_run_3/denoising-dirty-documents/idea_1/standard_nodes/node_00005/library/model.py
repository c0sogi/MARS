import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from library.config import (
    PATCH_SIZE,
    MODEL_WEIGHTS_PATH,
    WORKING_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
)


class SimpleDnCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image denoising.
    Cite solution_lesson_node_00001: Uses non-linear, spatially adaptive methods (CNN)
    to better handle edge preservation compared to global linear filters.
    """

    def __init__(self):
        super(SimpleDnCNN, self).__init__()
        # Input: 1 channel (grayscale)
        # Output: 1 channel (denoised)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Predict the clean image directly
        return self.features(x)


class DenoisingModel:
    """
    Wrapper for the PyTorch DnCNN model to maintain compatibility with the training workflow.
    """

    def __init__(self, patch_size=PATCH_SIZE):
        self.patch_size = patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleDnCNN().to(self.device)

    def fit(self, X, y):
        """
        Trains the CNN on provided patch data.

        Args:
            X (np.ndarray): Input patches (N, 1, H, W).
            y (np.ndarray): Target patches (N, 1, H, W).
        """
        print(f"Training DenoisingModel (CNN) on {self.device}...")

        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), MODEL_WEIGHTS_PATH)
        print(f"Model weights saved to {MODEL_WEIGHTS_PATH}")

    def load_weights(self):
        """
        Loads learned weights from the cache directory.
        """
        if os.path.exists(MODEL_WEIGHTS_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(MODEL_WEIGHTS_PATH, map_location=self.device)
                )
                return True
            except Exception as e:
                print(f"Error loading weights: {e}")
                return False
        return False

    def predict(self, image):
        """
        Applies the learned CNN to a full image to remove noise.

        Args:
            image (np.ndarray): Normalized input image (H, W).

        Returns:
            np.ndarray: Denoised image (H, W).
        """
        # Ensure model weights are available if not trained in this session
        if not hasattr(self, "model"):
            self.model = SimpleDnCNN().to(self.device)

        self.model.eval()

        # Prepare input tensor: (1, 1, H, W)
        img_tensor = (
            torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self.model(img_tensor)

        # Convert back to numpy (H, W)
        denoised = output.squeeze().cpu().numpy()

        # Clip values
        denoised = np.clip(denoised, 0.0, 1.0)

        return denoised
