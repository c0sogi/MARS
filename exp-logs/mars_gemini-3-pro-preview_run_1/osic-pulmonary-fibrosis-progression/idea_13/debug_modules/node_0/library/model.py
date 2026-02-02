import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Import from provided libraries
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.dataset import get_dataloaders


class DualAxisNet(nn.Module):
    def __init__(self, tabular_input_dim, embed_dim=512, pretrained=True):
        super(DualAxisNet, self).__init__()

        # 1. Independent Visual Backbones (EfficientNet-B0)
        # num_classes=0 returns the pooled feature vector directly
        self.backbone_ax = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )

        # Get feature dimension of EfficientNet-B0 (usually 1280)
        backbone_dim = self.backbone_ax.num_features

        # Projections to shared embedding dimension
        self.proj_ax = nn.Linear(backbone_dim, embed_dim)
        self.proj_cor = nn.Linear(backbone_dim, embed_dim)

        # 2. Tabular Embedding (MLP)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 3. Learnable Readout Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 4. Symmetric Attention Fusion
        # batch_first=True: (Batch, Seq, Feature)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )

        # 5. Regression Head
        # Input: Updated [CLS] + Original Tabular Embedding (Residual Anchor)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 3),  # alpha (slope), sigma_base, sigma_growth
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.cls_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_ax, img_cor, tab, meta):
        """
        Args:
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            tab: (B, Tabular_Dim)
            meta: (B, 2) -> [Baseline_FVC, Weeks_From_Baseline]
        """
        batch_size = img_ax.size(0)

        # --- Feature Extraction ---
        # Visual
        feat_ax = self.backbone_ax(img_ax)  # (B, 1280)
        feat_cor = self.backbone_cor(img_cor)  # (B, 1280)

        vec_ax = self.proj_ax(feat_ax).unsqueeze(1)  # (B, 1, D)
        vec_cor = self.proj_cor(feat_cor).unsqueeze(1)  # (B, 1, D)

        # Tabular
        vec_tab = self.tabular_mlp(tab).unsqueeze(1)  # (B, 1, D)

        # --- Fusion ---
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)

        # Construct Sequence: [CLS, Axial, Coronal, Tabular]
        tokens = torch.cat((cls_tokens, vec_ax, vec_cor, vec_tab), dim=1)  # (B, 4, D)

        # Self-Attention
        # We only care about the output for the CLS token (index 0)
        attn_out, _ = self.attention(tokens, tokens, tokens)
        cls_out = attn_out[:, 0, :]  # (B, D)

        # --- Residual Prior Anchor ---
        # Concatenate updated CLS with raw tabular embedding
        combined = torch.cat((cls_out, vec_tab.squeeze(1)), dim=1)  # (B, 2*D)

        # --- Prediction ---
        params = self.head(combined)  # (B, 3)

        alpha = params[:, 0]
        sigma_base_logit = params[:, 1]
        sigma_growth_logit = params[:, 2]

        # Enforce positivity for sigma
        sigma_base = F.softplus(sigma_base_logit)
        sigma_growth = F.softplus(sigma_growth_logit)

        # Calculate FVC and Confidence
        # meta[:, 0] = Baseline_FVC
        # meta[:, 1] = Weeks_From_Baseline

        baseline_fvc = meta[:, 0]
        delta_t = meta[:, 1]

        fvc_pred = baseline_fvc + alpha * delta_t
        confidence = sigma_base + sigma_growth * torch.abs(delta_t)

        return fvc_pred, confidence


def criterion(fvc_true, fvc_pred, sigma):
    """
    Differentiable Modified Laplace Log Likelihood Loss.
    Retains clipping logic to handle outliers during training.
    Loss = -Metric
    """
    # Clip sigma at 70
    sigma_clipped = torch.clamp(sigma, min=70)

    # Calculate delta and clip at 1000
    delta = torch.abs(fvc_true - fvc_pred)
    delta_clipped = torch.clamp(delta, max=1000)

    # Metric calculation
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=fvc_true.device))
    metric = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
        sqrt_2 * sigma_clipped
    )

    # We want to maximize metric, so minimize -metric
    return -torch.mean(metric)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tab = batch["tab"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        fvc_pred, confidence = model(img_ax, img_cor, tab, meta)

        loss = criterion(target, fvc_pred, confidence)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_conf = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab = batch["tab"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            fvc_pred, confidence = model(img_ax, img_cor, tab, meta)

            all_true.extend(target.cpu().numpy())
            all_pred.extend(fvc_pred.cpu().numpy())
            all_conf.extend(confidence.cpu().numpy())

    score = laplace_log_likelihood_metric(
        np.array(all_true), np.array(all_pred), np.array(all_conf)
    )
    return score


def generate_submission(
    model, loader, device, output_path="./submission/submission.csv"
):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab = batch["tab"].to(device)
            meta = batch["meta"].to(device)
            patient_weeks = batch["patient_week"]

            fvc_pred, confidence = model(img_ax, img_cor, tab, meta)

            fvc_pred = fvc_pred.cpu().numpy()
            confidence = confidence.cpu().numpy()

            for i in range(len(patient_weeks)):
                # Metric requires confidence clipped at 70, but we output raw confidence
                # The metric calculation handles clipping, but submission usually expects raw
                # or clipped. The prompt says "confidence values are clipped at 70 ml...
                # The file should contain... FVC, Confidence".
                # Usually best to submit the value the model thinks is right, but
                # we will ensure it's not below 70 for safety if the model output lower.
                conf_val = max(confidence[i], 70)

                results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": fvc_pred[i],
                        "Confidence": conf_val,
                    }
                )

    df_sub = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_pipeline(epochs=20, batch_size=32, patience=6, lr=1e-4, seed=42):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_dir="./metadata",
        cache_dir="./working/idea_13/",
        batch_size=batch_size,
        num_workers=4,
        img_size=224,
    )

    # Determine tabular input dimension from a batch
    sample_batch = next(iter(train_loader))
    tabular_dim = sample_batch["tab"].shape[1]
    print(f"Tabular Feature Dimension: {tabular_dim}")

    # 2. Model
    model = DualAxisNet(tabular_input_dim=tabular_dim, embed_dim=512, pretrained=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 3. Training Loop
    best_score = -float("inf")
    best_model_path = "./working/best_model_idea_13.pth"
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.10f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print("  -> New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Best Validation Score: {best_score:.10f}")

    # 4. Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(
        model, test_loader, device, output_path="./submission/submission.csv"
    )
