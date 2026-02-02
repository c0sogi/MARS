import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    Wraps a base optimizer (e.g., AdamW) and performs a dual-step update to minimize
    both the loss value and the loss sharpness (landscape smoothness). This implementation
    is designed to stabilize training on micro-datasets and high-capacity models.

    References:
        Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters to optimize.
            base_optimizer: The class of the base optimizer (e.g., torch.optim.AdamW).
            rho (float): Neighborhood size (radius of the perturbation).
            adaptive (bool): Whether to use element-wise adaptive perturbation (ASAM).
            **kwargs: Keyword arguments passed to the base optimizer (e.g., lr, weight_decay).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer with the parameter groups
        # We pass self.param_groups to ensure the base optimizer manages the same groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync param_groups and defaults to ensure consistency
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step of SAM:
        1. Compute the global gradient norm.
        2. Calculate the perturbation epsilon based on the gradient and rho.
        3. Save the current weight values.
        4. Add epsilon to the weights (climb to the local maximum of loss).
        5. Optionally zero gradients to prepare for the second backward pass.

        Args:
            zero_grad (bool): If True, clears the gradients after the step.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the original parameter value to state
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation e_w
                # Standard SAM: e_w = rho * grad / grad_norm
                # Adaptive SAM: scales by parameter magnitude
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation: w_adv = w + e_w
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step of SAM:
        1. Restore the original weights (subtract epsilon).
        2. Perform the base optimizer step using the gradients computed at the perturbed state.

        Args:
            zero_grad (bool): If True, clears the gradients after the step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameter value
                p.data = self.state[p]["old_p"]

        # Update parameters using the gradients from the perturbed state
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step method is not supported for SAM in this implementation
        because it requires an external closure or manual control of the two passes.
        """
        raise NotImplementedError(
            "SAM requires manual control of the two forward-backward passes. "
            "Use `optimizer.first_step(zero_grad=True)` after the first backward, "
            "and `optimizer.second_step(zero_grad=True)` after the second backward."
        )

    def _grad_norm(self):
        """
        Calculates the global L2 norm of the gradients across all parameters.
        """
        # We assume all parameters are on the same device or compatible for norm calculation.
        # This is generally true for single-GPU training.
        shared_device = self.param_groups[0]["params"][0].device

        stack = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
            .norm(p=2)
            .to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if len(stack) == 0:
            return torch.tensor(0.0, device=shared_device)

        norm = torch.norm(torch.stack(stack), p=2)
        return norm
