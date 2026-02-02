import torch


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Wraps a base optimizer to simultaneously minimize loss value and loss sharpness.
    In the context of the Iceberg Classifier, this helps find a flat minimum,
    improving generalization and robustness against speckle noise.

    Reference: https://arxiv.org/abs/2010.01412
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters.
            base_optimizer: The class of the base optimizer (e.g., torch.optim.AdamW).
            rho (float): Neighborhood size for the ascent step.
            adaptive (bool): Whether to use Adaptive SAM (ASAM).
            **kwargs: Keyword arguments passed to the base optimizer (lr, weight_decay, etc.).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer with the parameter groups
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Link param_groups to ensure synchronization
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the ascent step:
        1. Calculates the gradient norm.
        2. Computes the perturbation epsilon.
        3. Adds epsilon to the weights (w_adv = w + epsilon).
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the current 'clean' weights
                self.state[p]["old_p"] = p.data.clone()

                # Compute perturbation
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
        Performs the descent step:
        1. Restores the original weights (w).
        2. Updates weights using the gradients computed at w_adv.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original weights
                p.data = self.state[p]["old_p"]

        # Update using the base optimizer (gradients are from the perturbed state)
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using SAM logic.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
                                Required for SAM to perform the dual passes.
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"

        # Ensure closure enables gradients
        closure = torch.enable_grad()(closure)

        # 1. First Forward-Backward Pass (at w)
        # Computes gradients \nabla L(w)
        loss = closure()

        # 2. Ascent Step
        # Moves weights to w + epsilon
        self.first_step(zero_grad=True)

        # 3. Second Forward-Backward Pass (at w + epsilon)
        # Computes gradients \nabla L(w + epsilon)
        closure()

        # 4. Descent Step
        # Restores w, then updates w using \nabla L(w + epsilon)
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the global L2 norm of the gradients across all parameters.
        """
        # Ensure computation happens on the correct device (using the first param's device)
        if not self.param_groups or not self.param_groups[0]["params"]:
            return torch.tensor(0.0)

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

        norm = torch.norm(torch.stack(norms), p=2)
        return norm

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
