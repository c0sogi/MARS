import torch


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Wraps a base optimizer (e.g., SGD, Adam) to simultaneously minimize loss value
    and loss sharpness, improving generalization.

    References:
        Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters to optimize.
            base_optimizer: The class of the base optimizer (e.g., torch.optim.Adam).
            rho (float): Neighborhood size (default: 0.05).
            adaptive (bool): If True, uses element-wise adaptive scaling (ASAM).
            **kwargs: Keyword arguments passed to the base optimizer.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync param_groups and defaults
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the ascent step: w -> w + epsilon.
        Computes the perturbation epsilon based on current gradients and adds it to weights.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save current parameters to restore later
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation: w = w + e_w
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the descent step.
        Restores original weights (w) and updates them using gradients computed at (w + epsilon).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameters
                p.data = self.state[p]["old_p"]

        # Update parameters using the base optimizer
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
                                Required for SAM to perform the two-phase update.
        """
        assert closure is not None, "SAM requires closure, but it was not provided."

        # Ensure closure enables gradients
        closure = torch.enable_grad()(closure)

        # 1. First Pass: Compute gradients at current w
        loss = closure()

        # 2. Ascent: Move to w + epsilon (and zero grads)
        self.first_step(zero_grad=True)

        # 3. Second Pass: Compute gradients at w + epsilon
        closure()

        # 4. Descent: Restore w and update using gradients from step 3
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the global norm of gradients across all parameters.
        Handles device placement to ensure stability.
        """
        # Use the device of the first parameter for calculation
        shared_device = self.param_groups[0]["params"][0].device

        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        Ensures the base optimizer's param_groups are synced.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
