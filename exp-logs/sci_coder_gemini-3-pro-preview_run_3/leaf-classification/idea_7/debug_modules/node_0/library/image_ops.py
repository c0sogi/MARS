import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from library.config import Config


class ImagePreprocessor:
    """
    Handles loading and preprocessing of binary leaf images for dual-stream analysis.
    Generates multi-view (rotated) versions of each image and applies specific
    transforms for DINOv2 (Global) and ConvNeXt (Local) backbones.
    """

    def __init__(self):
        # ImageNet normalization statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Stream A: DINOv2 (ViT-Large)
        # Resolution: 518x518 (Config.IMG_SIZE_DINO)
        # Purpose: Global geometry and shape priors
        # We use Bicubic interpolation for better quality upscaling/downscaling
        self.transform_dino = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(
                    (Config.IMG_SIZE_DINO, Config.IMG_SIZE_DINO),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Stream B: ConvNeXt Large
        # Resolution: 1024x1024 (Config.IMG_SIZE_CONVNEXT)
        # Purpose: Fine-grained margin serrations and texture
        self.transform_conv = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(
                    (Config.IMG_SIZE_CONVNEXT, Config.IMG_SIZE_CONVNEXT),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def load_and_preprocess(self, relative_path):
        """
        Reads an image, generates 4 canonical rotated views (0, 90, 180, 270),
        and applies the specific resizing and normalization transforms.

        Args:
            relative_path (str): Path to the image file relative to the input directory.
                                 e.g., "images/10.jpg"

        Returns:
            tuple: (dino_batch, conv_batch)
                dino_batch (torch.Tensor): Shape (4, 3, 518, 518)
                conv_batch (torch.Tensor): Shape (4, 3, 1024, 1024)
        """
        # Construct full file path
        full_path = os.path.join(Config.INPUT_DIR, relative_path)

        # Load image in grayscale (binary source)
        # Using OpenCV for robust image loading
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not load image at {full_path}")

        # Generate 4 canonical rotated views
        # This enforces rotation invariance by explicitly providing all orientations
        views = []

        # 0 degrees (Original)
        views.append(img)

        # 90 degrees clockwise
        views.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

        # 180 degrees
        views.append(cv2.rotate(img, cv2.ROTATE_180))

        # 270 degrees (90 counter-clockwise)
        views.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

        dino_tensors = []
        conv_tensors = []

        for v in views:
            # Convert grayscale to RGB by replicating channels
            # Models pre-trained on ImageNet expect 3 channels
            v_rgb = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)

            # Apply DINO transforms
            dino_t = self.transform_dino(v_rgb)
            dino_tensors.append(dino_t)

            # Apply ConvNeXt transforms
            conv_t = self.transform_conv(v_rgb)
            conv_tensors.append(conv_t)

        # Stack views into a batch
        # Output shapes: (4, 3, H, W)
        dino_batch = torch.stack(dino_tensors)
        conv_batch = torch.stack(conv_tensors)

        return dino_batch, conv_batch
