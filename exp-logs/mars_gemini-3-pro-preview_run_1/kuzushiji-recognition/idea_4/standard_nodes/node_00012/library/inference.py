import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_metadata, create_submission_file
from library.dataset import KuzushijiDataset
from library.model import CenterNetConvNeXt


class Predictor:
    """
    Handles the inference pipeline for the Kuzushiji character recognition task.
    """

    def __init__(self, checkpoint_name="best_model.pth"):
        """
        Args:
            checkpoint_name (str): Name of the model checkpoint file in the working directory.
        """
        self.device = Config.DEVICE
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, checkpoint_name)

        # 1. Load Test Metadata
        # Using the utility function to leverage caching if available
        self.test_df = load_metadata(Config.TEST_METADATA_PATH)

        # 2. Initialize Dataset
        # We initialize in 'test' mode. This also loads the unicode map to build the class index.
        self.dataset = KuzushijiDataset(self.test_df, mode="test")

        # Create inverse mapping (Index -> Unicode) for decoding predictions
        self.idx_to_char = {v: k for k, v in self.dataset.char_to_idx.items()}

        # Ensure Config.NUM_CLASSES matches the dataset's class count (derived from unicode map)
        Config.NUM_CLASSES = self.dataset.num_classes

        # 3. Initialize DataLoader
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # 4. Initialize Model
        self.model = CenterNetConvNeXt(num_classes=Config.NUM_CLASSES).to(self.device)
        self._load_weights()

    def _load_weights(self):
        """
        Loads the trained model weights.
        """
        if os.path.exists(self.checkpoint_path):
            print(f"Loading model weights from {self.checkpoint_path}")
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Checkpoint not found at {self.checkpoint_path}. Using random initialization."
            )

    def _nms(self, heatmap, kernel=3):
        """
        Performs Non-Maximum Suppression using Max Pooling.
        """
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heatmap, (kernel, kernel), stride=1, padding=pad
        )
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def _gather_feat(self, feat, ind):
        """
        Gathers features at specific indices.
        """
        dim = feat.size(1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.view(feat.size(0), dim, -1).permute(0, 2, 1)
        feat = feat.gather(1, ind)
        return feat

    def _decode(self, hm, wh, reg, cls_logits, K=1200):
        """
        Decodes model outputs into detections.
        """
        batch_size, _, height, width = hm.shape

        # Heatmap -> Sigmoid -> NMS
        hm = torch.sigmoid(hm)
        hm = self._nms(hm)

        # Find top K peaks
        hm = hm.view(batch_size, -1)
        scores, inds = torch.topk(hm, K)

        # Convert indices to grid coordinates
        ys = inds.div(width, rounding_mode="floor").float()
        xs = (inds % width).float()

        # Gather regression offsets
        reg = self._gather_feat(reg, inds)

        # Apply offsets to grid coordinates
        xs = xs.view(batch_size, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + reg[:, :, 1:2]

        # Gather classification predictions
        cls_feat = self._gather_feat(cls_logits, inds)
        clses = torch.argmax(cls_feat, dim=2).view(batch_size, K, 1)

        # Scale back to model input size (stride 4)
        xs = xs * 4
        ys = ys * 4

        scores = scores.view(batch_size, K, 1)

        # Concatenate results: [x, y, score, class]
        detections = torch.cat([xs, ys, scores, clses.float()], dim=2)

        return detections

    def run(self, output_path="./submission/submission.csv"):
        """
        Runs inference on the test set and generates the submission file.
        """
        self.model.eval()
        predictions = []

        print(f"Starting inference on {len(self.test_loader.dataset)} images...")

        with torch.no_grad():
            for batch in self.test_loader:
                imgs = batch["image"].to(self.device)
                img_ids = batch["image_id"]
                orig_hs = batch["orig_h"]
                orig_ws = batch["orig_w"]

                # Forward Pass
                outputs = self.model(imgs)

                # Decode
                dets = self._decode(
                    outputs["hm"],
                    outputs["wh"],
                    outputs["reg"],
                    outputs["cls"],
                    K=Config.MAX_DETECTIONS,
                )

                dets = dets.cpu().numpy()

                # Process batch results
                for i in range(len(img_ids)):
                    img_id = img_ids[i]
                    orig_h = orig_hs[i].item()
                    orig_w = orig_ws[i].item()
                    det = dets[i]

                    # Filter by confidence threshold
                    mask = det[:, 2] >= Config.CONF_THRESHOLD
                    det = det[mask]

                    # Coordinate Mapping: Model Input (1024x1024) -> Original Image
                    # The transform used is LongestMaxSize + PadIfNeeded (Center)
                    scale = Config.IMG_SIZE / max(orig_h, orig_w)
                    resized_w = orig_w * scale
                    resized_h = orig_h * scale

                    # Calculate padding that was added
                    pad_w = (Config.IMG_SIZE - resized_w) / 2
                    pad_h = (Config.IMG_SIZE - resized_h) / 2

                    label_strs = []
                    for d in det:
                        x_pred, y_pred, score, cls_idx = d

                        # 1. Remove Padding
                        x_unpad = x_pred - pad_w
                        y_unpad = y_pred - pad_h

                        # 2. Rescale
                        x_orig = x_unpad / scale
                        y_orig = y_unpad / scale

                        # 3. Clip to valid image range
                        x_orig = max(0, min(orig_w, x_orig))
                        y_orig = max(0, min(orig_h, y_orig))

                        # 4. Map Class Index to Unicode
                        cls_idx = int(cls_idx)
                        if cls_idx in self.idx_to_char:
                            char = self.idx_to_char[cls_idx]
                            # Format: Unicode X Y
                            label_strs.append(f"{char} {int(x_orig)} {int(y_orig)}")

                    label_str = " ".join(label_strs)
                    predictions.append({"image_id": img_id, "labels": label_str})

        # Save Submission
        print(f"Generating submission file with {len(predictions)} predictions.")
        create_submission_file(predictions, output_path=output_path)
        print("Inference complete.")
