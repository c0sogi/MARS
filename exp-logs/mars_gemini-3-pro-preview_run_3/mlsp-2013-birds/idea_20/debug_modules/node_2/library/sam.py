import torch
import torch.nn as nn
import torch.optim as optim


class SAM(optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.

    Wraps a base optimizer (e.g., SGD, AdamW) to minimize both the loss value
    and the loss sharpness, improving generalization.

    References:
        Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: The base optimizer class (e.g. torch.optim.AdamW).
            rho: Neighborhood size (default: 0.05).
            adaptive: Element-wise scaling of rho (ASAM) (default: False).
            **kwargs: Keyword arguments passed to the base optimizer (e.g. lr, weight_decay).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer with the parameter groups managed by this class
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Share the state dictionary so that checkpoints save/load correctly
        # This ensures momentum/buffers from the base optimizer are stored in self.state
        self.base_optimizer.state = self.state

        # Sync param_groups just in case
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Perform the ascent step: perturb weights to maximize loss in the neighborhood.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the original parameter value
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation (Ascent)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Perform the descent step: restore weights and update using gradients from perturbed state.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameter value
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Apply the base optimizer update using the current gradients (calculated at perturbed state)
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
                                Required for SAM.
        """
        assert closure is not None, "SAM requires closure, but it was not provided."
        closure = torch.enable_grad()(closure)

        # 1. Ascent Step: Perturb weights based on current gradients
        self.first_step(zero_grad=True)

        # 2. Re-evaluate loss and gradients at the perturbed state
        loss = closure()

        # 3. Descent Step: Restore weights and update using perturbed gradients
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the global gradient norm across all parameter groups.
        """
        # Put everything on the same device to allow stacking
        # We assume the first parameter's device is the target device
        shared_device = self.param_groups[0]["params"][0].device

        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad
                    if group["adaptive"]:
                        grad = grad * torch.abs(p)
                    norms.append(grad.norm(p=2).to(shared_device))

        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        Ensures the base optimizer shares the loaded state.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        self.base_optimizer.state = self.state
