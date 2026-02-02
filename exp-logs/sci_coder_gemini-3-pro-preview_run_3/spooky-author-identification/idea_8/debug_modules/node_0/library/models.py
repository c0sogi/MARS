import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from library.config import ModelConfig, PathConfig, TrainConfig
from library.utils import set_seed


class StylometricTransformer(nn.Module):
    """
    A Transformer-based model with hierarchical pooling and Multi-Task Learning (MTL).

    Features:
    1. Loads backbone (optionally from DAPT checkpoint).
    2. Hierarchical Pooling: Concatenates [CLS] tokens from the last 4 hidden layers.
    3. Classification Head: Predicts author.
    4. Auxiliary Head: Predicts stylometric features (Log-Char-Length, Punctuation Density).
    """

    def __init__(self, backbone_name, num_labels=3, mtl_head_dim=2):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_labels = num_labels

        # Determine path to load: DAPT model if exists, else base model
        safe_name = backbone_name.replace("/", "-")
        dapt_path = os.path.join(PathConfig.MLM_MODELS_DIR, f"mlm_{safe_name}")

        if os.path.exists(os.path.join(dapt_path, "config.json")):
            print(f"Loading DAPT weights from {dapt_path}")
            config = AutoConfig.from_pretrained(dapt_path)
            config.output_hidden_states = True
            self.backbone = AutoModel.from_pretrained(dapt_path, config=config)
        else:
            print(f"DAPT weights not found. Loading base model {backbone_name}")
            config = AutoConfig.from_pretrained(backbone_name)
            config.output_hidden_states = True
            self.backbone = AutoModel.from_pretrained(backbone_name, config=config)

        self.hidden_size = config.hidden_size

        # Feature dimension after concatenating last 4 layers
        if ModelConfig.USE_LAST_4_LAYERS:
            self.feature_dim = self.hidden_size * 4
        else:
            self.feature_dim = self.hidden_size

        self.dropout = nn.Dropout(ModelConfig.HIDDEN_DROPOUT)

        # Main Classification Head
        self.classifier = nn.Linear(self.feature_dim, num_labels)

        # Auxiliary Regression Head (MTL)
        self.aux_regressor = nn.Linear(self.feature_dim, mtl_head_dim)

        # Loss functions
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_aux = nn.MSELoss()

    def get_features(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if ModelConfig.USE_LAST_4_LAYERS:
            # outputs.hidden_states is a tuple of (layer_0, ..., layer_N)
            # We want the last 4 layers.
            # Shape of each layer: (batch_size, seq_len, hidden_size)
            # We take the [CLS] token (index 0)

            hidden_states = outputs.hidden_states
            # Extract last 4 layers
            last_4_layers = [hidden_states[-i] for i in range(1, 5)]

            # Extract [CLS] tokens: (batch, hidden)
            cls_tokens = [layer[:, 0, :] for layer in last_4_layers]

            # Concatenate: (batch, hidden * 4)
            features = torch.cat(cls_tokens, dim=1)
        else:
            # Standard pooler output or last hidden state CLS
            # Deberta V3 doesn't always have a pooler, so use last hidden state CLS
            features = outputs.last_hidden_state[:, 0, :]

        return features

    def forward(self, input_ids, attention_mask, labels=None, aux_targets=None):
        features = self.get_features(input_ids, attention_mask)
        features = self.dropout(features)

        logits = self.classifier(features)
        aux_preds = self.aux_regressor(features)

        loss = None
        loss_cls = None
        loss_aux = None

        if labels is not None:
            loss_cls = self.criterion_cls(logits, labels)
            loss = loss_cls

            if aux_targets is not None and ModelConfig.USE_MTL:
                loss_aux = self.criterion_aux(aux_preds, aux_targets)
                loss = loss + TrainConfig.LAMBDA_MTL * loss_aux

        return {
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_aux": loss_aux,
            "logits": logits,
            "aux_preds": aux_preds,
        }


class StatisticalPredictor:
    """
    Wrapper for the statistical branch of the ensemble.
    Manages Logistic Regression and Naive Bayes models trained on TF-IDF features.
    """

    def __init__(self, seed=42):
        self.seed = seed
        # CalibratedClassifierCV is used to ensure probability outputs are well-calibrated
        # though LR is naturally probabilistic, NB often needs calibration.
        # However, for simplicity and speed in this context, we use the base models directly
        # or with simple settings, as we blend them later.

        # Logistic Regression
        self.lr_model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            multi_class="ovr",
            random_state=seed,
            max_iter=1000,
        )

        # Multinomial Naive Bayes
        self.nb_model = MultinomialNB(alpha=0.01)

    def fit(self, X, y):
        """
        Fits both statistical models.
        """
        self.lr_model.fit(X, y)
        self.nb_model.fit(X, y)
        return self

    def predict_proba_individual(self, X):
        """
        Returns probabilities from both models separately.

        Returns:
            dict: {'lr': np.ndarray, 'nb': np.ndarray}
        """
        lr_proba = self.lr_model.predict_proba(X)
        nb_proba = self.nb_model.predict_proba(X)
        return {"lr": lr_proba, "nb": nb_proba}

    def predict_proba(self, X, weights={"lr": 0.5, "nb": 0.5}):
        """
        Returns weighted average probabilities.
        """
        probas = self.predict_proba_individual(X)
        return weights["lr"] * probas["lr"] + weights["nb"] * probas["nb"]


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights to maximize loss, flattening the loss landscape.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
        start_epoch=0,
        adv_step=1,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, inputs, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        self._attack_step()

        # Forward pass with perturbed weights
        # We assume inputs is a dict compatible with model(**inputs)
        # We need to handle mixed precision if scaler is present
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                loss = outputs["loss"]
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(**inputs)
            loss = outputs["loss"]
            loss.backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
