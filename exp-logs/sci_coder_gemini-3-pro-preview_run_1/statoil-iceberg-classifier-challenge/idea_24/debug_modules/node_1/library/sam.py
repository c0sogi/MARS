import torch
import torch.optim as optim


class SAM(optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Wraps a base optimizer to minimize both loss value and loss sharpness.
    Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization".
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: The class of the base optimizer (e.g., torch.optim.AdamW).
            rho (float): The neighborhood size (default: 0.05).
            adaptive (bool): Whether to use Adaptive SAM (ASAM) (default: False).
            **kwargs: Keyword arguments passed to the base_optimizer.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer using the parameter groups from the parent class
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync param_groups and defaults to ensure consistency
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM:
        1. Calculates the gradient norm.
        2. Computes the perturbation (epsilon).
        3. Saves the current weight values.
        4. Applies the perturbation to the weights.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the original parameter value to state
                self.state[p]["old_p"] = p.data.clone()

                # Compute perturbation e_w
                # For standard SAM: e_w = rho * grad / grad_norm
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
        Performs the second step of SAM:
        1. Restores the original weight values.
        2. Updates the weights using the base optimizer and the gradients
           computed at the perturbed state.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original parameter value: w
                p.data = self.state[p]["old_p"]

        # Update w using gradients from w_adv
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that re-evaluates the model and returns the loss.
                                Required for SAM to compute gradients at the perturbed state.
        """
        assert closure is not None, "SAM requires closure, but it was not provided."
        closure = torch.enable_grad()(closure)

        # 1. Ascent Step
        # Expects gradients at current 'w' to be already computed (via backward before step)
        self.first_step(zero_grad=True)

        # 2. Re-evaluate loss and gradients at perturbed weights 'w_adv'
        loss = closure()

        # 3. Descent Step
        # Restore 'w' and update using gradients from 'w_adv'
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Calculates the global L2 norm of the gradients.
        """
        # Collect norms from all parameters, ensuring they are on the same device for stacking
        # We use the device of the first parameter as the reference
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
            return torch.tensor(0.0).to(shared_device)

        norm = torch.norm(torch.stack(stack), p=2)
        return norm

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
