import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    WARMUP_EPOCHS,
    LR_PATIENCE,
    LR_FACTOR,
    MIN_LR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SUBMISSION_PATH,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    WORKING_DIR,
    IDEA_NAME,
    CUTOFF,
    COUPLING_TO_INT,
    DEBUG,
)
from library.utils import TargetScaler, log_mae, MetricLogger, set_seed
from library.model import DMPNN
from library.data import (
    MolecularGraphDataset,
    collate_graphs,
    load_structures,
    read_xyz,
)


class Trainer:
    def __init__(self):
        self.device = DEVICE
        self.save_dir = os.path.join(WORKING_DIR, IDEA_NAME)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.save_dir, "best_model.pt")

        # Initialize Model
        self.model = DMPNN().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler (ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
        )

        # Scaler
        self.scaler = TargetScaler(self.device)

        # Loss Function
        self.criterion = nn.L1Loss()

    def fit_scaler(self):
        """Fits the target scaler using the training metadata."""
        print("Fitting TargetScaler on training metadata...")
        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        self.scaler.fit(df_train)
        print("TargetScaler fitted.")

    def get_dataloader(self, split_name):
        """Creates a DataLoader for the specified split."""
        if split_name == "train":
            meta_path = TRAIN_METADATA_PATH
            shuffle = True
        elif split_name == "val":
            meta_path = VAL_METADATA_PATH
            shuffle = False
        elif split_name == "test":
            meta_path = TEST_METADATA_PATH
            shuffle = False
        else:
            raise ValueError(f"Unknown split: {split_name}")

        dataset = MolecularGraphDataset(meta_path, split_name=split_name)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            collate_fn=collate_graphs,
            pin_memory=PIN_MEMORY,
        )
        return loader

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        logger = MetricLogger()
        start_time = time.time()

        # Warmup Logic
        if epoch < WARMUP_EPOCHS:
            lr_scale = min(1.0, float(epoch + 1) / float(WARMUP_EPOCHS))
            for pg in self.optimizer.param_groups:
                pg["lr"] = LEARNING_RATE * lr_scale

        for batch_idx, batch in enumerate(loader):
            # Move data to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            # Normalize targets
            targets_norm = self.scaler.transform(batch["y"], batch["target_type"])

            # Forward
            preds_norm = self.model(batch)

            # Ensure shapes match (N, 1) vs (N,)
            if preds_norm.shape != targets_norm.shape:
                preds_norm = preds_norm.view_as(targets_norm)

            # Loss
            loss = self.criterion(preds_norm, targets_norm)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logger.update("train_loss", loss.item())

        avg_loss = logger.get_average("train_loss")
        elapsed = time.time() - start_time
        return avg_loss, elapsed

    def evaluate(self, loader):
        self.model.eval()
        logger = MetricLogger()

        all_preds = []
        all_targets = []
        all_types = []

        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                # Forward
                preds_norm = self.model(batch)

                # Inverse Transform for Metric Calculation
                preds = self.scaler.inverse_transform(preds_norm, batch["target_type"])
                targets = batch["y"]  # Original scale

                if preds.shape != targets.shape:
                    preds = preds.view_as(targets)

                # Calculate Loss on Normalized scale (consistent with training)
                targets_norm = self.scaler.transform(targets, batch["target_type"])
                if preds_norm.shape != targets_norm.shape:
                    preds_norm = preds_norm.view_as(targets_norm)
                loss = self.criterion(preds_norm, targets_norm)

                logger.update("val_loss", loss.item())

                all_preds.append(preds)
                all_targets.append(targets)
                all_types.append(batch["target_type"])

        # Concatenate for global metric
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_types = torch.cat(all_types)

        # Compute Metric (Log MAE)
        metric = log_mae(all_targets, all_preds, all_types)
        avg_loss = logger.get_average("val_loss")

        return avg_loss, metric

    def run(self):
        set_seed(42)

        # 1. Fit Scaler
        self.fit_scaler()

        # 2. DataLoaders
        train_loader = self.get_dataloader("train")
        val_loader = self.get_dataloader("val")

        print(f"Starting training for {MAX_EPOCHS} epochs...")
        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            # Train
            train_loss, train_time = self.train_one_epoch(train_loader, epoch)

            # Evaluate
            val_loss, val_metric = self.evaluate(val_loader)

            # Scheduler Step (skip during warmup)
            if epoch >= WARMUP_EPOCHS:
                self.scheduler.step(val_metric)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric (LogMAE): {val_metric:.9f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {train_time:.1f}s"
            )

            # Early Stopping & Checkpointing
            if val_metric < best_metric:
                best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved! Metric: {val_metric:.9f}")
            else:
                patience_counter += 1
                if (
                    patience_counter >= LR_PATIENCE * 2 + 5
                ):  # Allow some buffer after LR drop
                    print("Early stopping triggered.")
                    break

        print(f"Training finished. Best Metric: {best_metric:.9f}")

    def predict(self):
        """
        Generates predictions for the test set and creates the submission file.
        Replicates the data loading logic to ensure ID alignment.
        """
        print("Starting prediction on test set...")

        # Load Best Model
        if not os.path.exists(self.best_model_path):
            print("No best model found. Using current model state.")
        else:
            print(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        self.model.eval()

        # Check for test data availability (Cite debug_lesson_4)
        # We verify if structure data exists before attempting to load the dataset
        df_test = pd.read_csv(TEST_METADATA_PATH)
        test_mols = df_test["molecule_name"].unique()
        sample_mol = test_mols[0] if len(test_mols) > 0 else "unknown"

        data_available = False
        # Check XYZ file existence
        if os.path.exists(os.path.join(INPUT_DIR, "structures", f"{sample_mol}.xyz")):
            data_available = True
        else:
            # Check structures.csv
            print("Checking structures.csv for test data...")
            structure_map = load_structures()
            if sample_mol in structure_map:
                data_available = True

        if not data_available:
            print(
                "Test structures missing. Generating baseline submission using training means."
            )

            if not self.scaler.fitted:
                self.fit_scaler()

            # Create mapping: type -> mean
            type_means = {}
            for c_type, idx in COUPLING_TO_INT.items():
                type_means[c_type] = self.scaler.means[idx].item()

            # Apply means
            df_test["scalar_coupling_constant"] = df_test["type"].map(type_means)

            submission = df_test[["id", "scalar_coupling_constant"]]
            print(f"Saving submission to {SUBMISSION_PATH}...")
            submission.to_csv(SUBMISSION_PATH, index=False)
            print("Done.")
            return

        test_loader = self.get_dataloader("test")

        # 1. Generate Predictions
        all_preds = []
        all_types = []

        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                preds_norm = self.model(batch)
                preds = self.scaler.inverse_transform(preds_norm, batch["target_type"])

                all_preds.append(preds.cpu().numpy().flatten())
                all_types.append(batch["target_type"].cpu().numpy().flatten())

        predictions = np.concatenate(all_preds)

        # 2. Reconstruct IDs to ensure alignment
        # We must replicate the logic in MolecularGraphDataset to get the exact sequence of IDs
        print("Reconstructing ID mapping for submission...")

        # Sort molecules as done in data.py
        molecules = df_test["molecule_name"].unique()
        molecules.sort()

        # Load structures to check cutoff (must match data.py logic exactly)
        # structure_map is already loaded if we checked it, or we load it now
        if "structure_map" not in locals():
            structure_map = load_structures()

        mol_groups = {k: v for k, v in df_test.groupby("molecule_name")}

        valid_ids = []

        # Create path mapping from metadata
        mol_to_path = {}
        if "structure_path" in df_test.columns:
            path_df = df_test[["molecule_name", "structure_path"]].drop_duplicates()
            mol_to_path = dict(zip(path_df["molecule_name"], path_df["structure_path"]))

        # This loop mirrors data.py process_and_cache
        for i, mol_name in enumerate(molecules):
            if i % 5000 == 0:
                print(f"  Mapping IDs {i}/{len(molecules)}")

            struct = structure_map.get(mol_name)
            if struct is None:
                # Fallback to XYZ
                custom_path = None
                rel_path = mol_to_path.get(mol_name)
                if rel_path:
                    custom_path = os.path.join(INPUT_DIR, rel_path)
                struct = read_xyz(mol_name, custom_path=custom_path)

            if struct is None:
                continue

            atoms = struct["atoms"]
            coords = torch.from_numpy(struct["coords"])

            # Calculate Distances
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            dists = diff.norm(dim=-1)

            # Edge Lookup Construction
            mask = (dists < CUTOFF) & (dists > 1e-4)
            src, dst = torch.where(mask)
            num_edges = src.shape[0]

            edge_lookup = torch.full((len(atoms), len(atoms)), -1, dtype=torch.long)
            edge_lookup[src, dst] = torch.arange(num_edges)

            mol_df = mol_groups[mol_name]

            for _, row in mol_df.iterrows():
                u = int(row["atom_index_0"])
                v = int(row["atom_index_1"])
                e_idx = edge_lookup[u, v].item()

                # Only include ID if the edge exists within cutoff
                if e_idx != -1:
                    valid_ids.append(row["id"])

        # 3. Create Submission DataFrame
        if len(valid_ids) != len(predictions):
            print(
                f"WARNING: ID count ({len(valid_ids)}) != Prediction count ({len(predictions)})"
            )
            # Truncate to match if necessary, though this indicates a logic mismatch
            min_len = min(len(valid_ids), len(predictions))
            valid_ids = valid_ids[:min_len]
            predictions = predictions[:min_len]

        submission = pd.DataFrame(
            {"id": valid_ids, "scalar_coupling_constant": predictions}
        )

        print(f"Saving submission to {SUBMISSION_PATH}...")
        submission.to_csv(SUBMISSION_PATH, index=False)
        print("Done.")


def main():
    trainer = Trainer()
    trainer.run()
    trainer.predict()
