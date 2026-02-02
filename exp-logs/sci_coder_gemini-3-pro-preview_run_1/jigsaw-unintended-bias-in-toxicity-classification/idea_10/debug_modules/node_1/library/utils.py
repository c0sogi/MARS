import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from library.config import CFG


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JigsawMetrics:
    """
    Calculates the Jigsaw Unintended Bias Toxicity Classification metric.
    Metric = 0.25*Overall_AUC + 0.25*Subgroup_AUC_Mean + 0.25*BPSN_AUC_Mean + 0.25*BNSP_AUC_Mean
    Means are calculated using generalized mean with p=-5.
    """

    def __init__(self):
        self.identity_columns = CFG.identity_cols
        self.w_overall = 0.25
        self.w_subgroup = 0.25
        self.w_bpsn = 0.25
        self.w_bnsp = 0.25
        self.p_mean = -5

    def compute_auc(self, y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5

    def power_mean(self, series, p):
        # Avoid division by zero or empty lists
        if not series:
            return 0.0
        total = np.sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def get_final_metric(self, y_true_df, y_pred):
        """
        y_true_df: DataFrame containing 'target' and identity columns.
        y_pred: Predicted probabilities (array-like).
        """
        # Convert inputs to consistent formats
        y_pred = np.array(y_pred)

        # Create evaluation dataframe
        eval_df = y_true_df.copy()
        eval_df["prediction"] = y_pred

        # Binarize target (>= 0.5 is toxic)
        eval_df["binary_target"] = (eval_df[CFG.target_col] >= 0.5).astype(int)

        # Binarize identities (>= 0.5 is considered a mention)
        for col in self.identity_columns:
            eval_df[col] = (eval_df[col] >= 0.5).astype(bool)

        # 1. Overall AUC
        overall_auc = self.compute_auc(eval_df["binary_target"], eval_df["prediction"])

        # Calculate Bias AUCs
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []

        for identity in self.identity_columns:
            # Subgroup AUC: Restrict to examples mentioning identity
            subgroup_df = eval_df[eval_df[identity] == True]
            if len(subgroup_df) > 0 and subgroup_df["binary_target"].nunique() > 1:
                auc = self.compute_auc(
                    subgroup_df["binary_target"], subgroup_df["prediction"]
                )
            else:
                auc = 0.5
            subgroup_aucs.append(auc)

            # BPSN: Background Positive, Subgroup Negative
            # (Non-toxic & Identity) OR (Toxic & No Identity)
            bpsn_mask = (~eval_df["binary_target"].astype(bool) & eval_df[identity]) | (
                eval_df["binary_target"].astype(bool) & ~eval_df[identity]
            )
            bpsn_df = eval_df[bpsn_mask]
            if len(bpsn_df) > 0 and bpsn_df["binary_target"].nunique() > 1:
                auc = self.compute_auc(bpsn_df["binary_target"], bpsn_df["prediction"])
            else:
                auc = 0.5
            bpsn_aucs.append(auc)

            # BNSP: Background Negative, Subgroup Positive
            # (Toxic & Identity) OR (Non-toxic & No Identity)
            bnsp_mask = (eval_df["binary_target"].astype(bool) & eval_df[identity]) | (
                ~eval_df["binary_target"].astype(bool) & ~eval_df[identity]
            )
            bnsp_df = eval_df[bnsp_mask]
            if len(bnsp_df) > 0 and bnsp_df["binary_target"].nunique() > 1:
                auc = self.compute_auc(bnsp_df["binary_target"], bnsp_df["prediction"])
            else:
                auc = 0.5
            bnsp_aucs.append(auc)

        # Calculate Generalized Means
        fn_subgroup = self.power_mean(subgroup_aucs, self.p_mean)
        fn_bpsn = self.power_mean(bpsn_aucs, self.p_mean)
        fn_bnsp = self.power_mean(bnsp_aucs, self.p_mean)

        final_score = (
            self.w_overall * overall_auc
            + self.w_subgroup * fn_subgroup
            + self.w_bpsn * fn_bpsn
            + self.w_bnsp * fn_bnsp
        )

        return final_score, overall_auc, fn_subgroup, fn_bpsn, fn_bnsp


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs weights to maximize loss, flattening the loss landscape.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Perturbs the model weights in the direction of the gradient.
        Must be called after loss.backward().
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # Perturbation direction
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    # Projection to epsilon ball
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def restore(self):
        """
        Restores the original model weights.
        """
        self._restore()


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy of weights and provides methods to apply/restore them.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Initializes the shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Updates the shadow weights using the current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replaces model weights with shadow weights (for inference/eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restores original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
