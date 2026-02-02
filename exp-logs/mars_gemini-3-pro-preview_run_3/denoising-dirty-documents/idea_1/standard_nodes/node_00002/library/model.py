import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from library.config import (
    PATCH_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WORKING_DIR,
)

MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")


class SimpleDnCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image denoising.
    Maps noisy image patches to clean image patches.
    """

    def __init__(self):
        super(SimpleDnCNN, self).__init__()
        # 3-layer CNN
        # Layer 1: Extract features
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Layer 2: Non-linear mapping
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Layer 3: Reconstruction
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # Use Sigmoid to ensure output is in [0, 1] range
        return torch.sigmoid(x)


class CNNFilter:
    """
    Wrapper for the PyTorch CNN model to handle training and inference.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = SimpleDnCNN().to(self.device)

    def fit(self, X, y):
        """
        Trains the CNN on provided patch data.

        Args:
            X (np.ndarray): Input patches (N, 1, H, W).
            y (np.ndarray): Target patches (N, 1, H, W).
        """
        print("Training CNNFilter...")

        # Convert to PyTorch tensors
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(y)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()

        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # Save model
        os.makedirs(WORKING_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def load_weights(self):
        """Loads weights from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(MODEL_PATH, map_location=self.device)
                )
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

    def predict(self, image):
        """
        Applies the CNN to a full image.

        Args:
            image (np.ndarray): Normalized input image (H, W).

        Returns:
            np.ndarray: Denoised image (H, W).
        """
        # Ensure model is ready
        if not os.path.exists(MODEL_PATH) and not hasattr(self, "model"):
            raise RuntimeError("Model not trained or loaded.")

        self.model.eval()

        # Prepare input: (H, W) -> (1, 1, H, W)
        img_tensor = (
            torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self.model(img_tensor)

        # Convert back to numpy: (1, 1, H, W) -> (H, W)
        denoised = output.cpu().squeeze().numpy()

        return denoised
