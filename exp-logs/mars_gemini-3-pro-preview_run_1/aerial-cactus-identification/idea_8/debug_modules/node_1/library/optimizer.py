import torch
import torch.optim


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer (e.g., AdamW)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Share the param_groups and state with the base optimizer
        # This ensures that momentum/buffers from base_optimizer are saved
        # when calling SAM.state_dict()
        self.param_groups = self.base_optimizer.param_groups
        self.base_optimizer.state = self.state

        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM:
        1. Compute the gradient norm.
        2. Calculate perturbation epsilon.
        3. Save current weights.
        4. Apply perturbation (w -> w + e).
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save original parameters to restore later
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation
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
        Performs the second step of SAM:
        1. Restore original weights (w + e -> w).
        2. Update weights using gradients from the perturbed state (base_optimizer.step).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameters
                p.data = self.state[p]["old_p"]

        # Update parameters using the gradients computed at the perturbed state
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step function requiring a closure.
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the global gradient norm across all parameter groups.
        """
        # Ensure we can handle cases where params are on different devices (though unlikely here)
        # We use the device of the first parameter as the reference for the norm calculation
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
        Custom load_state_dict to ensure base_optimizer properties are maintained.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
