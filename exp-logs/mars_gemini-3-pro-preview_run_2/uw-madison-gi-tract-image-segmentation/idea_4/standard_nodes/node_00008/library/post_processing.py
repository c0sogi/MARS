import numpy as np
from scipy import ndimage


def clean_3d_volume(masks, connectivity=None):
    """
    Refines the raw 2D predictions using 3D context.
    Constructs a 3D volume from a stack of 2D masks (if input is a list) and
    applies Connected Component Analysis to retain only the largest connected
    object per class, thereby removing isolated noise artifacts.

    Args:
        masks (np.ndarray or list): A 3D numpy array of shape (Depth, Height, Width)
                                    or a list of 2D numpy arrays.
                                    The input should be a binary mask (0 and 1).
        connectivity (np.ndarray, optional): Structuring element that defines
                                             feature connections. Defaults to None
                                             (standard 3D connectivity).

    Returns:
        np.ndarray: The cleaned 3D binary mask with the same shape and dtype as input.
    """
    # 1. Construct 3D volume if input is a list of 2D arrays
    if isinstance(masks, list):
        if not masks:
            return np.array([], dtype=np.uint8)
        volume = np.stack(masks, axis=0)
    else:
        volume = masks

    # Ensure input is treated as binary for labeling
    # We assume the input is for a single class (binary)
    binary_mask = volume > 0

    # If the mask is empty, return original
    if not np.any(binary_mask):
        return volume

    # 2. Apply 3D Connected Component Analysis
    # structure=None defaults to a connectivity of 3 (26-neighbors in 3D)
    labeled_vol, num_features = ndimage.label(binary_mask, structure=connectivity)

    # If there's more than one component, keep only the largest
    if num_features > 1:
        # Calculate the size (volume) of each component
        # The labels are 1, 2, ..., num_features
        component_sizes = ndimage.sum(
            binary_mask, labeled_vol, range(1, num_features + 1)
        )

        # Find the index of the largest component
        # argmax returns 0-based index, so add 1 to match label
        largest_label = np.argmax(component_sizes) + 1

        # Create a new binary mask containing only the largest component
        cleaned_bool = labeled_vol == largest_label

        # Cast back to original dtype (e.g., uint8)
        cleaned_volume = cleaned_bool.astype(volume.dtype)
        return cleaned_volume

    # If 1 or 0 components, return as is (already clean)
    return volume
