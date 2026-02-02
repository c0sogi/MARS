import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from library.config import Config


def load_image(path: str) -> torch.Tensor:
    """
    Loads an image from a file path, converts it to RGB, resizes it to the
    configured size, and converts it to a PyTorch tensor.

    Args:
        path: The file path to the image.

    Returns:
        torch.Tensor: A tensor of shape (3, H, W) with values in the range [0, 1].

    Raises:
        FileNotFoundError: If the image cannot be read from the path.
    """
    # Read image using OpenCV
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to the target dimensions specified in Config
    # We use INTER_AREA for shrinking which is generally better for downsampling
    img = cv2.resize(
        img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )

    # Convert to Tensor
    # TF.to_tensor converts (H, W, C) [0, 255] -> (C, H, W) [0.0, 1.0]
    tensor = TF.to_tensor(img)

    return tensor


def generate_rotated_views(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Generates a batch of equidistant rotated views for a given image tensor.

    This function creates 36 views (0, 10, ..., 350 degrees) to cover the
    full rotational manifold.

    Args:
        image_tensor: Input image tensor of shape (C, H, W) with values in [0, 1].

    Returns:
        torch.Tensor: A batch tensor of shape (36, C, H, W).
    """
    views = []

    # Generate angles: 0, 10, 20, ..., 350
    angles = np.linspace(0, 360, num=Config.NUM_ROTATIONS, endpoint=False)

    # The dataset consists of black leaves on white backgrounds.
    # Since the tensor is normalized to [0, 1], white corresponds to 1.0.
    # We must fill the corners introduced by rotation with white.
    fill_value = [1.0] * image_tensor.shape[0]

    for angle in angles:
        # Rotate the image
        # We use bilinear interpolation for better quality on the edges
        rotated_view = TF.rotate(
            image_tensor,
            angle=float(angle),
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=False,
            fill=fill_value,
        )
        views.append(rotated_view)

    # Stack all views into a single batch tensor
    batch = torch.stack(views, dim=0)

    return batch
