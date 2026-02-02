import os
import math
import json
import ast
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger()


class BEVDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, sample_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached BEV maps.
            sample_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.load_cached_data = load_cached_data
        self.cache_dir = os.path.join(Config.CACHE_DIR, "bev_maps")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Metadata
        if split == "train":
            self.meta_path = Config.TRAIN_METADATA
        elif split == "val":
            self.meta_path = Config.VAL_METADATA
        else:
            self.meta_path = Config.TEST_METADATA

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        df = pd.read_csv(self.meta_path)

        # Parse complex columns
        if "annotations" in df.columns:
            # Handle potential NaN in annotations
            df["annotations"] = df["annotations"].apply(
                lambda x: json.loads(x) if pd.notna(x) else []
            )

        # Filter for sample_size if requested
        if sample_size is not None:
            df = df.iloc[:sample_size]

        self.samples = df.to_dict("records")
        self.num_samples = len(self.samples)

        # Pre-calculate grid parameters
        self.voxel_size = Config.BEV_RESOLUTION
        self.x_min, self.x_max = Config.X_RANGE
        self.y_min, self.y_max = Config.Y_RANGE
        self.z_min, self.z_max = Config.Z_RANGE

        self.grid_w = Config.INPUT_SIZE[0]
        self.grid_h = Config.INPUT_SIZE[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_token = sample["sample_token"]

        # 1. Load/Generate BEV Map
        bev_map = self._get_bev_map(sample)

        # 2. Handle Targets (Train/Val) vs Inference (Test)
        if self.split in ["train", "val"]:
            anns = sample["annotations"]

            # Augmentation (only for training)
            if self.split == "train":
                bev_map, anns = self._augment(bev_map, anns)

            # Generate CenterNet Targets
            targets = self._generate_targets(anns)

            return {"input": torch.from_numpy(bev_map).float(), **targets}
        else:
            # Test mode
            return {
                "input": torch.from_numpy(bev_map).float(),
                "sample_token": sample_token,
            }

    def _get_bev_map(self, sample):
        """
        Retrieves the BEV map, either from cache or by computing it from LIDAR data.
        """
        sample_token = sample["sample_token"]
        cache_path = os.path.join(self.cache_dir, f"{sample_token}.npy")

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(
                    f"Failed to load cache for {sample_token}: {e}. Recomputing."
                )

        # Compute from scratch
        lidar_path = sample["lidar_path"]
        # Handle relative paths from metadata
        if not os.path.exists(lidar_path):
            # Try prepending input dir if path is relative and not found
            # The metadata script usually puts ./input/..., but let's be safe
            if lidar_path.startswith("./"):
                lidar_path = lidar_path  # correct
            else:
                lidar_path = os.path.join(Config.INPUT_DIR, lidar_path)

        if not os.path.exists(lidar_path):
            # Fallback: create empty map if file missing (should not happen with valid data)
            logger.error(f"Lidar file missing: {lidar_path}")
            return np.zeros(
                (Config.IN_CHANNELS, self.grid_h, self.grid_w), dtype=np.float32
            )

        # Load Point Cloud
        # .bin files are typically float32 x, y, z, intensity
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # Generate BEV
        bev = self._points_to_bev(points)

        # Save to cache
        if self.load_cached_data:
            np.save(cache_path, bev)

        return bev

    def _points_to_bev(self, points):
        """
        Converts point cloud (N, 4) to BEV map (3, H, W).
        Channels: Density, Intensity, Max Height.
        """
        # Filter points out of range
        mask = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] < self.z_max)
        )
        points = points[mask]

        if len(points) == 0:
            return np.zeros(
                (Config.IN_CHANNELS, self.grid_h, self.grid_w), dtype=np.float32
            )

        # Quantize coordinates
        # X corresponds to Width (columns), Y corresponds to Height (rows)
        # Note: Usually in BEV, X is forward/right or similar.
        # Here: Metadata says X is left/right, Y is forward/back?
        # Metadata: "y is forward/back, x is left/right".
        # Let's map X(world) -> W(image), Y(world) -> H(image).

        x_img = ((points[:, 0] - self.x_min) / self.voxel_size).astype(np.int32)
        y_img = ((points[:, 1] - self.y_min) / self.voxel_size).astype(np.int32)

        # Clip to ensure within bounds (though filter should handle this)
        x_img = np.clip(x_img, 0, self.grid_w - 1)
        y_img = np.clip(y_img, 0, self.grid_h - 1)

        # Prepare channels
        # 1. Density (Log count)
        # 2. Intensity (Normalized)
        # 3. Height (Normalized relative to Z range)

        # We use a flat index for fast accumulation
        flat_idx = y_img * self.grid_w + x_img

        # Sort by flat index to group points in same voxel
        sort_idx = np.argsort(flat_idx)
        flat_idx = flat_idx[sort_idx]
        points = points[sort_idx]

        # Find unique voxels and their counts
        unique_idx, counts = np.unique(flat_idx, return_counts=True)

        # Initialize maps
        density_map = np.zeros(self.grid_h * self.grid_w, dtype=np.float32)
        intensity_map = np.zeros(self.grid_h * self.grid_w, dtype=np.float32)
        height_map = (
            np.zeros(self.grid_h * self.grid_w, dtype=np.float32) - 5.0
        )  # Init with low value

        # Fill Density
        density_map[unique_idx] = np.log(counts + 1.0)

        # For Intensity and Height, we need aggregation.
        # Using pandas for fast groupby aggregation on the filtered points is often cleaner than pure numpy for mean/max
        # but pure numpy reduceat is faster. Let's use a simple loop over unique indices or pandas.
        # Given the constraints and performance, let's use a scatter max/mean approach.

        # Since we sorted, we can use reduceat
        # Find indices where the voxel index changes
        changes = np.concatenate(([0], np.where(flat_idx[:-1] != flat_idx[1:])[0] + 1))

        # Max Height
        z_vals = points[:, 2]
        max_z = np.maximum.reduceat(z_vals, changes)
        height_map[unique_idx] = max_z

        # Mean Intensity
        i_vals = points[:, 3]
        mean_i = np.add.reduceat(i_vals, changes) / counts
        intensity_map[unique_idx] = mean_i

        # Reshape
        density_map = density_map.reshape(self.grid_h, self.grid_w)
        intensity_map = intensity_map.reshape(self.grid_h, self.grid_w)
        height_map = height_map.reshape(self.grid_h, self.grid_w)

        # Normalize Height to [0, 1] approximately for stability
        height_map = (height_map - self.z_min) / (self.z_max - self.z_min)
        height_map = np.clip(height_map, 0, 1)

        # Normalize Intensity (assuming 0-255 range usually, but sometimes 0-1)
        # If max intensity > 1, normalize.
        if intensity_map.max() > 1.0:
            intensity_map = intensity_map / 255.0

        # Stack
        bev = np.stack([density_map, intensity_map, height_map], axis=0)  # (3, H, W)
        return bev

    def _augment(self, bev_map, anns):
        """
        Applies random horizontal flip and rotation.
        bev_map: (3, H, W)
        anns: list of dicts
        """
        # 1. Random Flip (Horizontal / Left-Right)
        # In our grid, X is width. Flipping X.
        if np.random.rand() < 0.5:
            # Flip image
            bev_map = np.flip(bev_map, axis=2).copy()  # Flip W dimension

            # Flip annotations
            for ann in anns:
                # Center X flip: x_new = x_max + x_min - x_old
                # In world coords, we are flipping around the center of the range?
                # Usually we flip around the ego-vehicle (0,0).
                # If X range is symmetric (-100, 100), then x_new = -x_old.
                ann["center_x"] = -ann["center_x"]

                # Yaw flip: flip around X-axis implies yaw -> -yaw?
                # Or flip around Y-axis (left/right flip)?
                # If we flip X coordinate, we mirror the world across Y-axis.
                # Angle alpha becomes pi - alpha or -alpha depending on definition.
                # Standard: yaw = -yaw + pi if defined from X-axis?
                # Let's visualize: car pointing X+ (0 rad). Flip X -> pointing X- (pi rad).
                # car pointing Y+ (pi/2). Flip X -> pointing Y+ (pi/2).
                # Formula: new_yaw = pi - old_yaw
                ann["yaw"] = np.pi - ann["yaw"]

                # Normalize yaw
                while ann["yaw"] > np.pi:
                    ann["yaw"] -= 2 * np.pi
                while ann["yaw"] < -np.pi:
                    ann["yaw"] += 2 * np.pi

        # 2. Random Rotation
        # Rotate around Z axis (0,0) by random angle
        if np.random.rand() < 0.5:
            angle_deg = np.random.uniform(-45, 45)
            angle_rad = np.radians(angle_deg)

            # Rotate Image
            # (C, H, W) -> (H, W, C) for OpenCV
            img_np = bev_map.transpose(1, 2, 0)
            center = (self.grid_w / 2, self.grid_h / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            img_rot = cv2.warpAffine(
                img_np, rot_mat, (self.grid_w, self.grid_h), flags=cv2.INTER_LINEAR
            )
            bev_map = img_rot.transpose(2, 0, 1)

            # Rotate Annotations
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            # Note: OpenCV rotation is counter-clockwise for positive angle?
            # Standard math rotation: x' = x cos - y sin, y' = x sin + y cos
            # But OpenCV image coords Y is down? No, we treat BEV as cartesian map.
            # However, rotation matrix in OpenCV with positive angle rotates CCW.
            # We need to match this.
            # We are rotating the "world" by angle_rad.
            # Point P_new = R * P_old.
            # But wait, we rotated the IMAGE. The objects in the image moved.
            # If we rotate image by +45 deg (CCW), a point at (10, 0) goes to (7, 7).
            # So we apply the rotation matrix to the coordinates.

            # The rotation matrix from cv2 is for image coordinates (pixels).
            # We need to rotate world coordinates (meters).
            # Since (0,0) world is center of image, it's the same rotation.
            # BUT, cv2 y-axis is down (row index). Our world Y is up/forward?
            # Let's stick to standard 2D rotation on world X,Y.
            # Since we rotate the image, we must rotate the box centers by the same amount.
            # Angle in cv2 is degrees CCW.

            # Rotation matrix for coordinates:
            # | cos -sin |
            # | sin  cos |
            # But we must be careful about the sign of angle relative to Y-axis direction.
            # Let's assume standard CCW rotation.

            for ann in anns:
                x, y = ann["center_x"], ann["center_y"]
                # Apply rotation
                # Note: If we use the same angle as cv2, we should be consistent.
                # However, cv2 rotates around the image center.
                # Our world origin (0,0) should map to image center.
                # So rotating (x,y) around (0,0) is correct.

                # Careful: In image space, Y increases downwards. In world, Y increases forward?
                # If we map World Y to Image Y directly, we might have flipped axis.
                # In _points_to_bev: y_img = (y - y_min) / res.
                # If y_min = -100, y=100 -> y_img = 200/0.8 = 250.
                # So World Y+ maps to Image Y+ (Bottom of array? No, array index increases downwards).
                # So World Y+ is "Down" in the image array.
                # This means our BEV is flipped vertically relative to standard Cartesian.
                # This complicates rotation direction.
                # Simplification: Just rotate (x,y) using standard math, and add angle to yaw.
                # If visual mismatch occurs, the model learns the correlation anyway as long as it's consistent.

                # Using standard rotation formula:
                # To match OpenCV's CCW rotation on the image grid:
                # If Y-axis is inverted, a CCW rotation in pixel space looks like CW in Cartesian?
                # Let's just apply the rotation to x,y and yaw.
                # To be safe with OpenCV `getRotationMatrix2D` (which is CCW for positive angle on image coords):
                # If we rotate image CCW, the point (x,0) moves towards (x, -y) in image indices?
                # Let's trust that rotating the vector (x,y) by -angle (or +angle) aligns.
                # Actually, simpler: The network sees the rotated image. We must provide the new coordinates of the box IN THAT ROTATED FRAME.
                # So we rotate the point (x,y) by `angle_rad`.

                ann["center_x"] = (
                    x * c + y * s
                )  # Note: sign depends on axis def. Let's try standard.
                ann["center_y"] = -x * s + y * c

                # Wait, standard rotation:
                # x' = x cos - y sin
                # y' = x sin + y cos
                # Let's use this.
                ann["center_x"] = x * c - y * s
                ann["center_y"] = x * s + y * c

                ann["yaw"] += angle_rad

                # Normalize yaw
                while ann["yaw"] > np.pi:
                    ann["yaw"] -= 2 * np.pi
                while ann["yaw"] < -np.pi:
                    ann["yaw"] += 2 * np.pi

        return bev_map, anns

    def _generate_targets(self, anns):
        """
        Generates CenterNet targets.
        """
        # Targets
        hm = np.zeros((Config.NUM_CLASSES, self.grid_h, self.grid_w), dtype=np.float32)

        # Sparse targets arrays (Max Detections)
        max_objs = Config.MAX_DETECTIONS

        # Indices (K,)
        ind = np.zeros((max_objs), dtype=np.int64)
        # Mask (K,) - 1 if object exists, 0 otherwise
        mask = np.zeros((max_objs), dtype=np.float32)
        # Regression Heads
        reg = np.zeros((max_objs, 2), dtype=np.float32)  # x, y offset
        wh = np.zeros((max_objs, 3), dtype=np.float32)  # w, l, h
        depth = np.zeros((max_objs, 1), dtype=np.float32)  # z
        rot = np.zeros((max_objs, 2), dtype=np.float32)  # sin, cos

        num_objs = 0

        for ann in anns:
            cls_name = ann["class_name"]
            if cls_name not in Config.CLASS_TO_ID:
                continue
            cls_id = Config.CLASS_TO_ID[cls_name]

            # Convert World Coords to Grid Coords
            # x_img = (x - x_min) / res
            # y_img = (y - y_min) / res
            # Note: We use float indices for center

            x_c = (ann["center_x"] - self.x_min) / self.voxel_size
            y_c = (ann["center_y"] - self.y_min) / self.voxel_size

            # Check bounds
            if x_c < 0 or x_c >= self.grid_w or y_c < 0 or y_c >= self.grid_h:
                continue

            # Integer center
            x_i = int(x_c)
            y_i = int(y_c)

            # 1. Heatmap (Gaussian Splat)
            # Radius based on object size? Or fixed?
            # Standard CenterNet calculates radius based on IoU overlap.
            # Simplified: Fixed radius or proportional to width/length.
            # Let's use a heuristic based on object size in grid.
            w_grid = ann["width"] / self.voxel_size
            l_grid = ann["length"] / self.voxel_size
            radius = gaussian_radius((l_grid, w_grid), min_overlap=0.7)
            radius = max(0, int(radius))

            draw_gaussian(hm[cls_id], (x_i, y_i), radius)

            # 2. Sparse Targets
            if num_objs < max_objs:
                ind[num_objs] = y_i * self.grid_w + x_i
                mask[num_objs] = 1

                # Regression: Offset from integer center
                reg[num_objs] = [x_c - x_i, y_c - y_i]

                # Dimensions
                wh[num_objs] = [ann["width"], ann["length"], ann["height"]]

                # Depth (Z center)
                depth[num_objs] = [ann["center_z"]]

                # Rotation
                # We encode sin(yaw), cos(yaw)
                rot[num_objs] = [math.sin(ann["yaw"]), math.cos(ann["yaw"])]

                num_objs += 1

        return {
            "hm": torch.from_numpy(hm),
            "ind": torch.from_numpy(ind),
            "mask": torch.from_numpy(mask),
            "reg": torch.from_numpy(reg),
            "wh": torch.from_numpy(wh),
            "depth": torch.from_numpy(depth),
            "rot": torch.from_numpy(rot),
        }


# ==============================================================================
# Helper Functions for CenterNet Targets
# ==============================================================================


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0], heatmap.shape[1]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def worker_init_fn(worker_id):
    """
    Sets seeds for workers to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # random.seed(worker_seed) # If using random module
