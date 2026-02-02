import torch


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Wraps a base optimizer (e.g., AdamW) to simultaneously minimize loss value and loss sharpness.
    Reference: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters to optimize.
            base_optimizer: Class of the base optimizer (e.g. torch.optim.AdamW).
            rho: Neighborhood size for perturbation (default: 0.05).
            adaptive: Whether to use adaptive SAM (element-wise scaling) (default: False).
            **kwargs: Keyword arguments passed to the base optimizer (lr, weight_decay, etc.).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer with the same parameter groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Ensure param_groups are shared/synced
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the ascent step: computes perturbation (epsilon) and applies it to the weights.

        Args:
            zero_grad (bool): If True, zeroes the gradients after applying perturbation.
                              Typically True because we need to compute fresh gradients at w + epsilon.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Compute epsilon: e_w = rho * grad / grad_norm
                # Adaptive SAM scales by parameter magnitude
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation: w_adv = w + e_w
                p.add_(e_w)

                # Store epsilon in state to revert later
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the descent step: reverts perturbation and updates weights using base optimizer.

        Args:
            zero_grad (bool): If True, zeroes the gradients after the update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "e_w" in self.state[p]:
                    # Revert perturbation: w = w_adv - e_w
                    p.sub_(self.state[p]["e_w"])
                    # Clean up transient state
                    del self.state[p]["e_w"]

        # Update weights using the gradients computed at w_adv
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using SAM logic.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
                                Required for SAM to compute gradients at the perturbed state.
        """
        assert closure is not None, "SAM requires closure, but it was not provided."
        closure = torch.enable_grad()(closure)

        # 1. Ascent Step
        # Assumes gradients at current weights 'w' are already computed (e.g. via backward() before step())
        self.first_step(zero_grad=True)

        # 2. Compute gradients at perturbed weights 'w + epsilon'
        loss = closure()

        # 3. Descent Step
        # Reverts weights to 'w' and updates them using the gradients from step 2
        self.second_step()

        return loss

    def _grad_norm(self):
        """Computes the global L2 norm of the gradients across all parameters."""
        # Ensure computation is on the correct device (handle model parallelism if necessary)
        # We use the device of the first parameter found.
        shared_device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    shared_device = p.device
                    break
            if shared_device is not None:
                break

        if shared_device is None:
            return torch.tensor(0.0)

        stack = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
            .norm(p=2)
            .to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if len(stack) == 0:
            return torch.tensor(0.0).to(shared_device)

        norm = torch.norm(torch.stack(stack), p=2)
        return norm

    def load_state_dict(self, state_dict):
        """Proxies load_state_dict to the base optimizer."""
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups

    def state_dict(self):
        """Proxies state_dict to the base optimizer."""
        return self.base_optimizer.state_dict()
