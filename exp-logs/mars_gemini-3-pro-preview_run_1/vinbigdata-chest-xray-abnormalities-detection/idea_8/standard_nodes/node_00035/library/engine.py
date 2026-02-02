import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm.auto import tqdm

from library.config import Config
from library.utils import calculate_map, seed_everything
from library.loss import CenterNetLoss


class Engine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CenterNetLoss(
            hm_weight=1.0, wh_weight=0.1, off_weight=1.0, global_weight=1.0
        )
        self.best_map = 0.0

    def _decode_detections(self, hm, wh, reg, K=100):
        """
        Decodes model outputs into bounding boxes.
        hm: (B, C, H, W)
        wh: (B, 2, H, W)
        reg: (B, 2, H, W)
        Returns: (B, K, 6) -> [xmin, ymin, xmax, ymax, score, class]
        """
        batch_size, C, H, W = hm.shape

        # 1. Heatmap NMS via MaxPool
        hm = torch.sigmoid(hm)
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        keep = (hmax == hm).float()
        hm = hm * keep

        # 2. Top K
        # Flatten to (B, C*H*W)
        hm_flat = hm.view(batch_size, -1)
        topk_scores, topk_inds = torch.topk(hm_flat, K)

        topk_clses = (topk_inds // (H * W)).float()
        topk_inds = topk_inds % (H * W)

        topk_ys = (topk_inds // W).float()
        topk_xs = (topk_inds % W).float()

        # 3. Retrieve Regressions at indices
        # wh: (B, 2, H*W)
        wh = wh.view(batch_size, 2, -1)
        # reg: (B, 2, H*W)
        reg = reg.view(batch_size, 2, -1)

        # Gather
        # indices must be expanded: (B, 2, K)
        inds_expand = topk_inds.unsqueeze(1).expand(batch_size, 2, K)

        wh_gathered = wh.gather(2, inds_expand)  # (B, 2, K)
        reg_gathered = reg.gather(2, inds_expand)  # (B, 2, K)

        w = wh_gathered[:, 0, :]
        h = wh_gathered[:, 1, :]

        off_x = reg_gathered[:, 0, :]
        off_y = reg_gathered[:, 1, :]

        # 4. Reconstruct Boxes (Feature Map Scale)
        # Center = (xs + off_x, ys + off_y)
        xs = topk_xs + off_x
        ys = topk_ys + off_y

        x1 = xs - w / 2
        y1 = ys - h / 2
        x2 = xs + w / 2
        y2 = ys + h / 2

        # 5. Stack [x1, y1, x2, y2, score, class]
        # Shape: (B, 6, K) -> permute to (B, K, 6)
        dets = torch.stack([x1, y1, x2, y2, topk_scores, topk_clses], dim=1)
        dets = dets.permute(0, 2, 1)

        return dets

    def train_one_epoch(self, loader, epoch, debug=False):
        self.model.train()
        running_loss = 0.0
        running_stats = {}

        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False, disable=None)

        for i, batch in enumerate(pbar):
            # Move to device
            imgs = batch["image"].to(self.device)
            targets = {
                "hm": batch["hm"].to(self.device),
                "wh": batch["wh"].to(self.device),
                "reg": batch["reg"].to(self.device),
                "ind": batch["ind"].to(self.device),
                "reg_mask": batch["reg_mask"].to(self.device),
                "global_label": batch["global_label"].to(self.device),
            }

            # Forward
            outputs = self.model(imgs)

            # Loss
            loss, stats = self.criterion(outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Logging
            running_loss += loss.item()
            for k, v in stats.items():
                running_stats[k] = running_stats.get(k, 0.0) + v.item()

            pbar.set_postfix(loss=loss.item())

            if debug and i >= 10:
                break

        avg_loss = running_loss / (i + 1)
        avg_stats = {k: v / (i + 1) for k, v in running_stats.items()}

        print(
            f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} "
            f"(HM: {avg_stats['hm_loss']:.4f}, WH: {avg_stats['wh_loss']:.4f}, "
            f"Off: {avg_stats['off_loss']:.4f}, Global: {avg_stats['global_loss']:.4f})"
        )

        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader, debug=False):
        self.model.eval()

        # Prepare GT lookup (image_id -> list of boxes/labels)
        # We need this because the batch only contains CenterNet targets, not raw boxes for mAP
        gt_lookup = {}
        # Access the underlying data list from the dataset
        raw_data = loader.dataset.data
        for item in raw_data:
            img_id = item["image_id"]
            # Filter out "No finding" (14) for mAP calculation
            valid = item["labels"] != Config.NO_FINDING_CLASS_ID
            gt_lookup[img_id] = {
                "boxes": item["boxes"][valid],
                "labels": item["labels"][valid],
            }

        pred_boxes_all = []
        pred_scores_all = []
        pred_labels_all = []
        gt_boxes_all = []
        gt_labels_all = []

        pbar = tqdm(loader, desc="Evaluating", leave=False, disable=None)

        for i, batch in enumerate(pbar):
            imgs = batch["image"].to(self.device)
            img_ids = batch["image_id"]
            orig_shapes = batch["original_shape"].numpy()  # (B, 2) [h, w]

            # Forward
            outputs = self.model(imgs)

            # Decode
            # dets: (B, K, 6)
            dets = self._decode_detections(outputs["hm"], outputs["wh"], outputs["reg"])

            # Global Head Gating
            # outputs['global_cls'] is (B, 1) logits
            global_probs = torch.sigmoid(outputs["global_cls"]).cpu().numpy().flatten()

            dets = dets.cpu().numpy()

            batch_size = dets.shape[0]
            for b in range(batch_size):
                img_id = img_ids[b]
                orig_h, orig_w = orig_shapes[b]

                # Get Predictions for this image
                img_dets = dets[b]  # (K, 6)

                # Gated Inference Logic
                # Model predicts P(Finding). Config.GLOBAL_THRESHOLD (0.8) is for "No Finding".
                # If P(No Finding) > 0.8 => 1 - P(Finding) > 0.8 => P(Finding) < 0.2
                p_finding = global_probs[b]
                if p_finding < (1.0 - Config.GLOBAL_THRESHOLD):
                    # Suppress all boxes
                    final_boxes = np.empty((0, 4))
                    final_scores = np.empty((0,))
                    final_labels = np.empty((0,))
                else:
                    # Filter by confidence threshold
                    mask = img_dets[:, 4] > Config.CONF_THRESHOLD
                    valid_dets = img_dets[mask]

                    if len(valid_dets) == 0:
                        final_boxes = np.empty((0, 4))
                        final_scores = np.empty((0,))
                        final_labels = np.empty((0,))
                    else:
                        # Rescale boxes
                        # Current: Feature map (160x160)
                        # Step 1: Scale to Input Size (640x640)
                        scale_feat = 4.0  # Stride
                        boxes_input = valid_dets[:, :4] * scale_feat

                        # Step 2: Scale to Original Size
                        # x * (orig_w / 640), y * (orig_h / 640)
                        sx = orig_w / Config.IMAGE_SIZE
                        sy = orig_h / Config.IMAGE_SIZE

                        boxes_orig = boxes_input.copy()
                        boxes_orig[:, 0] *= sx
                        boxes_orig[:, 2] *= sx
                        boxes_orig[:, 1] *= sy
                        boxes_orig[:, 3] *= sy

                        # Clip to image boundaries
                        boxes_orig[:, 0] = np.clip(boxes_orig[:, 0], 0, orig_w)
                        boxes_orig[:, 2] = np.clip(boxes_orig[:, 2], 0, orig_w)
                        boxes_orig[:, 1] = np.clip(boxes_orig[:, 1], 0, orig_h)
                        boxes_orig[:, 3] = np.clip(boxes_orig[:, 3], 0, orig_h)

                        final_boxes = boxes_orig
                        final_scores = valid_dets[:, 4]
                        final_labels = valid_dets[:, 5].astype(int)

                # Store Preds
                pred_boxes_all.append(final_boxes)
                pred_scores_all.append(final_scores)
                pred_labels_all.append(final_labels)

                # Store GTs
                if img_id in gt_lookup:
                    gt_data = gt_lookup[img_id]
                    gt_boxes_all.append(gt_data["boxes"])
                    gt_labels_all.append(gt_data["labels"])
                else:
                    # Should not happen in val, but safe fallback
                    gt_boxes_all.append(np.empty((0, 4)))
                    gt_labels_all.append(np.empty((0,)))

            if debug and i >= 10:
                break

        # Calculate mAP
        metrics = calculate_map(
            pred_boxes_all,
            pred_scores_all,
            pred_labels_all,
            gt_boxes_all,
            gt_labels_all,
            num_classes=Config.NUM_CLASSES,
            iou_threshold=Config.IOU_THRESHOLD,
        )

        print(f"Validation mAP: {metrics['mAP']:.10f}")
        return metrics["mAP"]

    @torch.no_grad()
    def predict_test(self, loader, output_path, debug=False):
        self.model.eval()
        results = []

        pbar = tqdm(loader, desc="Inference", leave=False, disable=None)

        for i, batch in enumerate(pbar):
            imgs = batch["image"].to(self.device)
            img_ids = batch["image_id"]
            orig_shapes = batch["original_shape"].numpy()

            outputs = self.model(imgs)

            dets = self._decode_detections(outputs["hm"], outputs["wh"], outputs["reg"])
            global_probs = torch.sigmoid(outputs["global_cls"]).cpu().numpy().flatten()
            dets = dets.cpu().numpy()

            batch_size = dets.shape[0]
            for b in range(batch_size):
                img_id = img_ids[b]
                orig_h, orig_w = orig_shapes[b]
                img_dets = dets[b]

                p_finding = global_probs[b]

                prediction_string = ""

                # Gated Inference
                if p_finding < (1.0 - Config.GLOBAL_THRESHOLD):
                    # Predict No Finding
                    prediction_string = "14 1 0 0 1 1"
                else:
                    # Filter confidence
                    mask = img_dets[:, 4] > Config.CONF_THRESHOLD
                    valid_dets = img_dets[mask]

                    if len(valid_dets) == 0:
                        # Fallback if global head says finding but no boxes found
                        # The prompt says: "If you predict that there are NO objects... predict 14 1 0 0 1 1"
                        # We can trust the box head here.
                        prediction_string = "14 1 0 0 1 1"
                    else:
                        # Rescale
                        scale_feat = 4.0
                        boxes_input = valid_dets[:, :4] * scale_feat

                        sx = orig_w / Config.IMAGE_SIZE
                        sy = orig_h / Config.IMAGE_SIZE

                        boxes_orig = boxes_input.copy()
                        boxes_orig[:, 0] *= sx
                        boxes_orig[:, 2] *= sx
                        boxes_orig[:, 1] *= sy
                        boxes_orig[:, 3] *= sy

                        # Clip
                        boxes_orig[:, 0] = np.clip(boxes_orig[:, 0], 0, orig_w)
                        boxes_orig[:, 2] = np.clip(boxes_orig[:, 2], 0, orig_w)
                        boxes_orig[:, 1] = np.clip(boxes_orig[:, 1], 0, orig_h)
                        boxes_orig[:, 3] = np.clip(boxes_orig[:, 3], 0, orig_h)

                        # Format String
                        preds_list = []
                        for k in range(len(valid_dets)):
                            cls_id = int(valid_dets[k, 5])
                            score = valid_dets[k, 4]
                            xmin, ymin, xmax, ymax = boxes_orig[k]

                            preds_list.append(
                                f"{cls_id} {score:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                            )

                        prediction_string = " ".join(preds_list)

                results.append(
                    {"image_id": img_id, "PredictionString": prediction_string}
                )

            if debug and i >= 5:
                break

        df_sub = pd.DataFrame(results)
        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    def run(
        self, train_loader, val_loader, test_loader, epochs=Config.EPOCHS, debug=False
    ):
        print(f"Starting training for {epochs} epochs on {self.device}...")

        saved_best_this_run = False

        for epoch in range(epochs):
            # Train
            self.train_one_epoch(train_loader, epoch, debug=debug)

            # Validate
            val_map = self.evaluate(val_loader, debug=debug)

            # Checkpoint
            if val_map > self.best_map:
                self.best_map = val_map
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with mAP: {val_map:.10f}")
                saved_best_this_run = True

            # Save Last
            last_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")
            torch.save(self.model.state_dict(), last_path)

            if debug:
                print("Debug mode: stopping after 1 epoch.")
                break

        print("Training completed.")

        # Load Best Model for Inference
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if saved_best_this_run and os.path.exists(best_path):
            print(f"Loading best model from {best_path}...")
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        else:
            print(
                "No new best model saved (or mAP did not improve). Using last model state."
            )

        # Generate Submission
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        self.predict_test(test_loader, sub_path, debug=debug)
