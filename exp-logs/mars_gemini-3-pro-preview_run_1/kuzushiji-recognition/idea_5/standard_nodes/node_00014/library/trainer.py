import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import KuzushijiDataset
from library.model import SwinCenterNet
from library.loss import CenterNetLoss
from library.utils import decode_center_net, calc_f1_score, _transpose_and_gather_feat


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = Config.DEVICE
        self.setup_seed(Config.SEED)

        # Directories
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Data Loaders
        print("Initializing Datasets...")
        train_dataset = KuzushijiDataset(
            split="train", debug_size=Config.DEBUG_SAMPLE_SIZE if debug else None
        )
        val_dataset = KuzushijiDataset(
            split="val", debug_size=Config.DEBUG_SAMPLE_SIZE if debug else None
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Model
        print("Initializing Model...")
        self.model = SwinCenterNet()
        self.model.to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
        )

        self.criterion = CenterNetLoss()

        # History
        self.best_f1 = -1.0
        self.log_path = Config.LOG_PATH

        # Initialize log file
        with open(self.log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_f1,time_sec\n")

    def setup_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch["image"])
            loss, stats = self.criterion(outputs, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                outputs = self.model(batch["image"])
                loss, stats = self.criterion(outputs, batch)
                running_loss += loss.item()

                # Decode Predictions
                # outputs['hm']: (B, 1, H/4, W/4)
                preds = decode_center_net(
                    outputs["hm"],
                    outputs["wh"],
                    outputs["reg"],
                    K=Config.MAX_DETECTIONS,
                )
                # preds: (B, K, 6) -> [x, y, w, h, score, class]
                # Coordinates are in output stride scale. Convert to 1024x1024.
                preds[:, :, :4] *= Config.OUTPUT_STRIDE

                # Reconstruct Ground Truth from dense maps for F1 calculation
                # We need [class, x, y, w, h] in 1024x1024 scale

                # Gather GT values at object indices
                ind = batch["ind"]
                mask = batch["mask"]

                # wh and reg are dense (B, 2, H, W). Gather them.
                wh_gt = _transpose_and_gather_feat(batch["wh"], ind)  # (B, K, 2)
                reg_gt = _transpose_and_gather_feat(batch["reg"], ind)  # (B, K, 2)
                cls_gt = batch["cls_ids"]  # (B, K)

                # Calculate coordinates
                # ind is index in flattened (H/4 * W/4) map
                W_out = outputs["hm"].shape[3]
                ys = (ind // W_out).float()
                xs = (ind % W_out).float()

                # Apply regression offset
                xs = xs + reg_gt[:, :, 0]
                ys = ys + reg_gt[:, :, 1]

                # Scale to image size
                stride = Config.OUTPUT_STRIDE
                cx = xs * stride
                cy = ys * stride
                w = wh_gt[:, :, 0] * stride
                h = wh_gt[:, :, 1] * stride

                # Convert center to top-left for metric compatibility
                x_tl = cx - w / 2
                y_tl = cy - h / 2

                # Stack to (B, K, 5) -> [class, x, y, w, h]
                # Note: calc_f1_score expects list of numpy arrays

                batch_size = preds.size(0)
                for i in range(batch_size):
                    # Filter predictions by score (optional, but good practice)
                    p = preds[i].cpu().numpy()

                    # Filter targets by mask
                    valid_mask = mask[i].bool().cpu()
                    t_cls = cls_gt[i][valid_mask].cpu().float()
                    t_x = x_tl[i][valid_mask].cpu()
                    t_y = y_tl[i][valid_mask].cpu()
                    t_w = w[i][valid_mask].cpu()
                    t_h = h[i][valid_mask].cpu()

                    t = torch.stack([t_cls, t_x, t_y, t_w, t_h], dim=1).numpy()

                    all_preds.append(p)
                    all_targets.append(t)

        val_loss = running_loss / len(self.val_loader)
        f1, precision, recall = calc_f1_score(all_preds, all_targets)

        return val_loss, f1

    def fit(self):
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
        patience = 5
        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(epoch)
            val_loss, val_f1 = self.validate()
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val F1: {val_f1:.10f} | "
                f"Time: {elapsed:.2f}s"
            )

            # Log
            with open(self.log_path, "a") as f:
                f.write(f"{epoch},{train_loss},{val_loss},{val_f1},{elapsed}\n")

            # Save Best Model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New Best F1! Model saved to {Config.BEST_MODEL_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    def predict(self):
        print("Starting prediction on test set...")

        # Load Best Model
        if os.path.exists(Config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()

        # Load Test Data
        test_dataset = KuzushijiDataset(
            split="test", debug_size=Config.DEBUG_SAMPLE_SIZE if self.debug else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Unicode Map for decoding class indices
        df_uni = pd.read_csv(Config.UNICODE_MAP_PATH)
        # Assuming unicode_translation.csv has 'Unicode' column or first column
        if "Unicode" in df_uni.columns:
            chars = df_uni["Unicode"].values
        else:
            chars = df_uni.iloc[:, 0].values
        idx_to_char = {i: c for i, c in enumerate(chars)}

        results = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                image_ids = batch["image_id"]
                orig_sizes = batch["orig_size"].numpy()  # (B, 2) -> [H, W]

                outputs = self.model(images)

                preds = decode_center_net(
                    outputs["hm"],
                    outputs["wh"],
                    outputs["reg"],
                    K=Config.MAX_DETECTIONS,
                )

                # Process batch
                preds = preds.cpu().numpy()

                for i, img_id in enumerate(image_ids):
                    # Filter by confidence
                    p_det = preds[i]
                    # Sort by score descending
                    p_det = p_det[np.argsort(-p_det[:, 4])]

                    # Filter threshold
                    valid_indices = p_det[:, 4] >= Config.CONF_THRESHOLD
                    p_det = p_det[valid_indices]

                    # Limit max detections
                    if len(p_det) > Config.MAX_DETECTIONS:
                        p_det = p_det[: Config.MAX_DETECTIONS]

                    # Scale coordinates
                    # Current coords are in output stride (1/4 of 1024)
                    # Need to scale to 1024 first, then to original image size

                    orig_h, orig_w = orig_sizes[i]
                    scale_x = orig_w / Config.IMG_SIZE[1]
                    scale_y = orig_h / Config.IMG_SIZE[0]

                    label_strs = []

                    for det in p_det:
                        # det: [x, y, w, h, score, class]
                        # x, y are top-left in feature map space
                        # Convert to center in feature map space
                        x_c_feat = det[0]
                        y_c_feat = det[1]

                        # Note: decode_center_net returns (x + reg), (y + reg).
                        # These are centers in feature map space.
                        # Wait, let's verify decode_center_net in utils.py
                        # xs = xs.view(...) + reg[...]
                        # xs comes from topk_xs which are indices % width.
                        # So xs, ys are indeed center coordinates in feature map space.

                        # Scale to 1024x1024
                        x_c_1024 = x_c_feat * Config.OUTPUT_STRIDE
                        y_c_1024 = y_c_feat * Config.OUTPUT_STRIDE

                        # Scale to original image
                        final_x = int(x_c_1024 * scale_x)
                        final_y = int(y_c_1024 * scale_y)

                        cls_idx = int(det[5])
                        if cls_idx in idx_to_char:
                            char = idx_to_char[cls_idx]
                            # Format: Unicode X Y
                            label_strs.append(f"{char} {final_x} {final_y}")

                    # Join all labels
                    labels_str = " ".join(label_strs)
                    results.append({"image_id": img_id, "labels": labels_str})

        # Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
