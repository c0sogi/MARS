import numpy as np
from library import config, utils


class BBoxHandler:
    """
    Handles loading of MegaDetector bounding boxes and applying context-aware cropping logic.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the handler by loading the bounding box data.

        Args:
            load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                     Passed to utils.get_megadetector_boxes.
        """
        # Load DataFrame using the provided utility which handles caching (JSON -> Parquet)
        self.df = utils.get_megadetector_boxes(load_cached_data=load_cached_data)

        # Create an efficient lookup dictionary: image_id -> [x, y, w, h]
        # We use 'list' orientation to get the values directly
        self.bbox_map = self.df.set_index("image_id")[["x", "y", "w", "h"]].T.to_dict(
            "list"
        )

    def get_expanded_bbox(self, image_id, margin=config.CROP_MARGIN):
        """
        Retrieves the bounding box for a specific image, expands it by the margin,
        and clamps it to the image boundaries [0, 1].

        Args:
            image_id (str): The unique identifier for the image.
            margin (float): The percentage to expand the box (e.g., 0.2 for 20%).

        Returns:
            tuple: (x, y, w, h) in normalized coordinates [0, 1].
                   Returns (0.0, 0.0, 1.0, 1.0) if the image_id is not found.
        """
        # Default to full image if no detection exists for this ID
        if image_id not in self.bbox_map:
            return 0.0, 0.0, 1.0, 1.0

        # Retrieve raw box
        x, y, w, h = self.bbox_map[image_id]

        # Calculate center of the box
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Expand width and height by the margin
        # The margin is added to the total size (e.g., w * 1.2)
        w_new = w * (1.0 + margin)
        h_new = h * (1.0 + margin)

        # Calculate new top-left coordinates based on center
        x_new = cx - w_new / 2.0
        y_new = cy - h_new / 2.0

        # Clamp coordinates to [0, 1] range
        # We calculate x1, y1 (top-left) and x2, y2 (bottom-right)
        x1 = max(0.0, x_new)
        y1 = max(0.0, y_new)
        x2 = min(1.0, x_new + w_new)
        y2 = min(1.0, y_new + h_new)

        # Recalculate width and height after clamping
        w_final = x2 - x1
        h_final = y2 - y1

        return x1, y1, w_final, h_final
