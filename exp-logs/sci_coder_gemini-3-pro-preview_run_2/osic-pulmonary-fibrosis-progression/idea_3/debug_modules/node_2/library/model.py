import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import Config
from library.utils import laplace_log_likelihood, create_submission


class AttentionPooling(nn.Module):
    """
    Aggregates K slice embeddings into a single patient embedding using attention.
    """

    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (Batch, K_slices, Embed_Dim)
        # weights shape: (Batch, K_slices, 1)
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)

        # Weighted sum: (Batch, Embed_Dim)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class VaryingCoeffNet(nn.Module):
    """
    Attention-Enhanced Neural Varying-Coefficient Network.
    Predicts linear trajectory parameters (alpha, beta) and uncertainty (delta).
    """

    def __init__(self, tab_dim=8):
        super(VaryingCoeffNet, self).__init__()

        self.img_embed_dim = Config.EMBED_DIM
        self.tab_dim = tab_dim
        self.hidden_dim = Config.HIDDEN_DIM

        # 1. Attention Pooling for Image Features
        self.pool = AttentionPooling(self.img_embed_dim)

        # Fusion Dimension
        fusion_dim = self.img_embed_dim + self.tab_dim

        # 2. Trajectory Head (Predicts Intercept alpha and Slope beta)
        self.traj_mlp = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2),  # Output: [Intercept, Slope]
        )

        # 3. Uncertainty Head (Predicts Delta / MAE)
        # Takes same static embedding but is independent of Weeks
        self.unc_mlp = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),  # Output: [Delta]
        )

    def forward(self, img_feats, tab_feats, weeks):
        """
        img_feats: (B, K, 1280)
        tab_feats: (B, 8)
        weeks: (B, 1) or (B,)
        """
        # Ensure weeks is (B, 1)
        if weeks.dim() == 1:
            weeks = weeks.view(-1, 1)

        # 1. Pool Images
        img_emb = self.pool(img_feats)  # (B, 1280)

        # 2. Concatenate with Tabular
        combined = torch.cat([img_emb, tab_feats], dim=1)  # (B, 1288)

        # 3. Predict Trajectory Parameters
        traj_params = self.traj_mlp(combined)
        alpha = traj_params[:, 0:1]
        beta = traj_params[:, 1:2]

        # Linear Trajectory: FVC = alpha + beta * weeks
        fvc_pred = alpha + beta * weeks

        # 4. Predict Uncertainty (Delta)
        # Enforce positivity with Softplus
        delta_pred = F.softplus(self.unc_mlp(combined))

        return fvc_pred, delta_pred


def train_model(model, train_loader, val_loader):
    """
    Executes the sequential training strategy:
    Phase 1: Train FVC Trajectory (L1 Loss).
    Phase 2: Train Uncertainty Head (L1 Loss on Residuals).
    """
    device = Config.DEVICE
    model = model.to(device)

    print(f"Starting training on {device}...")

    # ==========================================
    # Phase 1: Train Trajectory (FVC)
    # ==========================================
    print("\n=== Phase 1: Training FVC Trajectory ===")

    # Freeze Uncertainty Head, Train Trajectory + Attention
    for param in model.unc_mlp.parameters():
        param.requires_grad = False
    for param in model.traj_mlp.parameters():
        param.requires_grad = True
    for param in model.pool.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            imgs, tabs, weeks, targets = [x.to(device) for x in batch]

            optimizer.zero_grad()
            fvc_pred, _ = model(imgs, tabs, weeks)

            # L1 Loss for FVC
            loss = F.l1_loss(fvc_pred.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation (FVC MAE)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, tabs, weeks, targets = [x.to(device) for x in batch]
                fvc_pred, _ = model(imgs, tabs, weeks)
                loss = F.l1_loss(fvc_pred.squeeze(), targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered for Phase 1.")
                break

    # Load best FVC model
    model.load_state_dict(best_model_state)

    # ==========================================
    # Phase 2: Train Uncertainty
    # ==========================================
    print("\n=== Phase 2: Training Uncertainty Head ===")

    # Freeze Trajectory + Attention, Train Uncertainty
    for param in model.traj_mlp.parameters():
        param.requires_grad = False
    for param in model.pool.parameters():
        param.requires_grad = False
    for param in model.unc_mlp.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    # Reset Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_metric = -float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Pre-compute residuals for training set?
    # No, we compute them on the fly using the frozen trajectory head.

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            imgs, tabs, weeks, targets = [x.to(device) for x in batch]

            optimizer.zero_grad()

            # Get predictions (Trajectory is frozen)
            with torch.no_grad():
                fvc_pred, _ = model(imgs, tabs, weeks)
                residuals = torch.abs(targets - fvc_pred.squeeze())

            # Predict Delta
            # We need to run forward again or partial forward to get gradients for unc_mlp
            # Since forward computes both, and traj is frozen, this is fine.
            _, delta_pred = model(imgs, tabs, weeks)

            # Loss: L1 between predicted Delta and actual Residual
            loss = F.l1_loss(delta_pred.squeeze(), residuals)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation (Laplace Metric)
        model.eval()
        all_true = []
        all_pred = []
        all_sigma = []

        with torch.no_grad():
            for batch in val_loader:
                imgs, tabs, weeks, targets = [x.to(device) for x in batch]
                fvc_pred, delta_pred = model(imgs, tabs, weeks)

                # Analytical Scaling: Sigma = Delta * sqrt(2)
                sigma_pred = delta_pred * np.sqrt(2)

                all_true.extend(targets.cpu().numpy())
                all_pred.extend(fvc_pred.cpu().numpy().flatten())
                all_sigma.extend(sigma_pred.cpu().numpy().flatten())

        val_metric = laplace_log_likelihood(all_true, all_pred, all_sigma)

        scheduler.step(val_metric)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Residual MAE: {train_loss:.4f} | Val Metric: {val_metric:.5f}"
        )

        if val_metric > best_metric:
            best_metric = val_metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered for Phase 2.")
                break

    model.load_state_dict(best_model_state)
    print(f"Training Complete. Best Validation Metric: {best_metric:.5f}")
    return model


def predict_and_submit(model, test_loader):
    """
    Generates predictions for the test set and creates the submission file.
    """
    device = Config.DEVICE
    model = model.to(device)
    model.eval()

    all_ids = []
    all_fvc = []
    all_conf = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            # Test loader returns: img, tab, weeks, patient_week_ids
            imgs, tabs, weeks, ids = batch
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            weeks = weeks.to(device)

            fvc_pred, delta_pred = model(imgs, tabs, weeks)

            # Analytical Scaling: Sigma = Delta * sqrt(2)
            sigma_pred = delta_pred * np.sqrt(2)

            all_ids.extend(ids)
            all_fvc.extend(fvc_pred.cpu().numpy().flatten())
            all_conf.extend(sigma_pred.cpu().numpy().flatten())

    # Create submission
    create_submission(all_ids, all_fvc, all_conf)
