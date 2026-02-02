import torch


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Wraps a base optimizer (e.g., SGD, AdamW) and performs a two-step update
    to minimize both the loss value and the loss sharpness.

    References:
        - Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters to optimize.
            base_optimizer: The class of the base optimizer (e.g. torch.optim.AdamW).
            rho (float): Neighborhood size (default: 0.05).
            adaptive (bool): Whether to use Adaptive SAM (ASAM).
            **kwargs: Keyword arguments passed to the base_optimizer (e.g. lr, weight_decay).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Instantiate the base optimizer using the parameter groups created by the parent class
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Ensure param_groups are synchronized
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step of SAM:
        1. Compute the gradient norm.
        2. Calculate the perturbation (epsilon).
        3. Save the current weight values.
        4. Add perturbation to weights to move to the neighborhood maximum.

        Args:
            zero_grad (bool): If True, clears gradients after the step.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save original parameter value
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation e_w
                # For standard SAM: e_w = rho * grad / grad_norm
                # For Adaptive SAM: e_w = rho * (p^2 * grad) / grad_norm (simplified concept)
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step of SAM:
        1. Restore the original weight values.
        2. Update the weights using the gradients computed at the perturbed point.

        Args:
            zero_grad (bool): If True, clears gradients after the step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameter value
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Update weights using the base optimizer (using gradients from the perturbed state)
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step method is not supported for SAM as it requires two distinct
        forward-backward passes controlled by the training loop.
        """
        raise NotImplementedError(
            "SAM requires manual usage of first_step() and second_step()."
        )

    def _grad_norm(self):
        """
        Calculates the global norm of the gradients.
        """
        # Put everything on the same device for calculation (assuming model fits on one device or handling explicitly)
        # We use the device of the first parameter found.
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

    def zero_grad(self, set_to_none=False):
        """
        Clears the gradients of all optimized parameters.
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Returns the state of the base optimizer.
        """
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the state into the base optimizer.
        """
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
