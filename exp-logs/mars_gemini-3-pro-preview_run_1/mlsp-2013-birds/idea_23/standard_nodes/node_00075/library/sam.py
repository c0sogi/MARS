import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.

    Wraps a base optimizer (e.g., SGD, AdamW) to simultaneously minimize loss value
    and loss sharpness, improving generalization.

    References:
        Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: Class of the base optimizer (e.g. torch.optim.AdamW).
            rho: Neighborhood size (default: 0.05).
            adaptive: If True, uses Adaptive SAM (ASAM) (default: False).
            **kwargs: Keyword arguments passed to the base optimizer (e.g. lr, weight_decay).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer with the parameter groups created by super().__init__
        # This ensures both optimizers point to the same parameter objects.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync defaults
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM: Gradient Ascent.

        Computes the perturbation 'epsilon' that maximizes the loss within the neighborhood rho,
        and applies it to the weights.

        Args:
            zero_grad (bool): If True, clears the gradients after the step.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Calculate perturbation e_w
                # For standard SAM: e_w = rho * grad / norm
                # For ASAM: e_w = rho * (p^2 * grad) / norm (simplified concept)
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation: w_adv = w + e_w
                p.add_(e_w)

                # Save e_w to state for restoration in second_step
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the second step of SAM: Gradient Descent.

        Restores the original weights and applies the base optimizer update using the
        gradients computed at the perturbed state.

        Args:
            zero_grad (bool): If True, clears the gradients after the step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original weights: w = w_adv - e_w
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])

        # Update weights using the gradients from the perturbed state
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step method compatible with closures.
        Performs the full SAM update cycle.
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        Calculates the global L2 norm of the gradients across all parameters.
        """
        # Identify device for calculation (assumes model is on one device or DDP handles sync)
        shared_device = self.param_groups[0]["params"][0].device

        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
            .norm(p=2)
            .to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if not norms:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    def zero_grad(self, set_to_none=False):
        """
        Clears the gradients of all optimized parameters.
        """
        self.base_optimizer.zero_grad(set_to_none)

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
