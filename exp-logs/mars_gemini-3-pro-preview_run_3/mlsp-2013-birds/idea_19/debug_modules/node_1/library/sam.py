import torch


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    References:
        - "Sharpness-Aware Minimization for Efficiently Improving Generalization" (Foret et al., 2021)
        - https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters to optimize.
            base_optimizer: The class of the base optimizer (e.g., torch.optim.AdamW).
            rho (float): The neighborhood size parameter (default: 0.05).
            adaptive (bool): Whether to use Adaptive SAM (ASAM) (default: False).
            **kwargs: Keyword arguments passed to the base_optimizer.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync param_groups
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM: Gradient Ascent.
        Computes the perturbation epsilon and applies it to the weights.

        Args:
            zero_grad (bool): Whether to zero gradients after the step.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the current parameter state
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation
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
        Performs the second step of SAM: Gradient Descent.
        Restores the original weights and applies the base optimizer update
        using the gradients computed at the perturbed state.

        Args:
            zero_grad (bool): Whether to zero gradients after the step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameter state: w = w_adv - e_w
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Update parameters using the base optimizer
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using a closure.
        This allows SAM to be used with standard training loops that support closures.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        assert closure is not None, "SAM requires closure, but it was not provided."
        closure = torch.enable_grad()(closure)

        # Step 1: Ascent
        self.first_step(zero_grad=True)

        # Compute loss at perturbed state
        loss = closure()

        # Step 2: Descent
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the norm of the gradients for the perturbation scaling.
        """
        # Collect all gradients that are not None
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # Apply adaptive scaling if enabled
                    g = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    grads.append(g.norm(p=2))

        if not grads:
            return torch.tensor(0.0)

        # Stack and compute global norm
        # Ensure all are on the same device for the stack operation
        device = grads[0].device
        norm = torch.norm(torch.stack([g.to(device) for g in grads]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
