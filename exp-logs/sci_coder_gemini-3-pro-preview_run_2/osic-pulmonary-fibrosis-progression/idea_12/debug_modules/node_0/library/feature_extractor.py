import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from library.config import Config
from library.utils import seed_everything


class EfficientNetExtractor:
    """
    Wraps a pre-trained EfficientNet-B0 model to extract features from medical images.
    Specifically designed to process the 4-image set (MIP + 3 Zonal Axials) per patient.
    """

    def __init__(self):
        """
        Initialize the model, load weights, and prepare for inference.
        """
        # Ensure reproducibility
        seed_everything(Config.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained EfficientNet-B0
        # Using default weights (ImageNet)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.model = models.efficientnet_b0(weights=weights)

        # Remove the classification head to extract embeddings
        # The classifier is usually a Sequential(Dropout, Linear)
        # Replacing with Identity returns the output of the avgpool layer (flattened)
        # Output dimension for B0 is 1280
        self.model.classifier = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # Define normalization transform (ImageNet stats)
        # Input images are already [0, 1] float32 from image_processing.py
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Converts a single numpy image (H, W) to a normalized tensor (1, 3, H, W).
        """
        # Convert numpy [0, 1] to tensor
        # Input shape: (H, W)
        tensor = torch.from_numpy(image).float()

        # Add channel dimension: (1, H, W)
        tensor = tensor.unsqueeze(0)

        # Expand to 3 channels (Grayscale -> RGB): (3, H, W)
        tensor = tensor.expand(3, -1, -1)

        # Normalize using ImageNet stats
        tensor = self.normalize(tensor)

        # Add batch dimension: (1, 3, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor

    def extract_features(self, images_dict: dict) -> np.ndarray:
        """
        Extracts and concatenates features for a single patient's image set.

        Args:
            images_dict (dict): Dictionary containing the 4 processed images.
                                Keys: 'mip', 'axial_1', 'axial_2', 'axial_3'.
                                Values: Numpy arrays of shape (IMAGE_SIZE, IMAGE_SIZE).

        Returns:
            np.ndarray: A flattened feature vector.
                        Shape: (4 * 1280,) = (5120,).
        """
        # Define the specific order of images for consistency
        keys = ["mip", "axial_1", "axial_2", "axial_3"]

        batch_tensors = []

        for key in keys:
            if key not in images_dict:
                raise ValueError(f"Missing key {key} in images_dict")

            img = images_dict[key]

            # Preprocess to (1, 3, H, W)
            tensor = self._preprocess_image(img)
            batch_tensors.append(tensor)

        # Concatenate into a single batch: (4, 3, H, W)
        batch = torch.cat(batch_tensors, dim=0)
        batch = batch.to(self.device)

        # Inference
        with torch.no_grad():
            # Forward pass
            # Output shape: (4, 1280)
            features = self.model(batch)

        # Move to CPU and convert to numpy
        features_np = features.cpu().numpy()

        # Flatten into a single vector
        # (4, 1280) -> (5120,)
        flattened_features = features_np.flatten()

        return flattened_features
