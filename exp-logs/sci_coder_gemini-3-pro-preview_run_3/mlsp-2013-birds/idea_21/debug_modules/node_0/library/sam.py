import torch


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Wraps a base optimizer to minimize both loss value and loss sharpness.
    Reference: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer
        # We pass self.param_groups so they share the same parameter groups structure
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync param_groups and defaults to ensure consistency
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step of SAM: Perturb weights to the neighborhood that maximizes loss.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the original parameter value
                self.state[p]["old_p"] = p.data.clone()

                # Calculate the perturbation e_w
                # For ASAM (Adaptive), scale by p^2, otherwise 1.0
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Apply perturbation: w -> w + e_w
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step of SAM: Restore weights and update using gradients from the perturbed state.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameter value: w + e_w -> w
                p.data = self.state[p]["old_p"]

        # Update parameters using the gradients computed at the perturbed state
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
                                Required for SAM to compute gradients at the perturbed state.
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # Ensure closure runs with autograd enabled

        # 1. Perturb weights based on current gradients (from the first backward pass)
        self.first_step(zero_grad=True)

        # 2. Recompute loss and gradients at the perturbed state
        loss = closure()

        # 3. Restore weights and update using the new gradients
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the global norm of gradients across all parameter groups.
        """
        # Identify the device of the first parameter with a gradient
        shared_device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    shared_device = p.device
                    break
            if shared_device is not None:
                break

        # If no gradients are present, return 0
        if shared_device is None:
            return torch.tensor(0.0)

        # Collect norms from all parameters
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
            .norm(p=2)
            .to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        # Compute global norm
        return torch.norm(torch.stack(norms), p=2)
