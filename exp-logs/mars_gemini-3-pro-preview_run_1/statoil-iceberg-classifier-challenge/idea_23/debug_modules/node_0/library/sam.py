import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    Wraps a base optimizer to perform the dual-pass optimization strategy:
    1. Perturb weights to maximize loss (ascent).
    2. Compute gradient at perturbed state.
    3. Update original weights using this gradient (descent).

    References:
        Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Parameters to optimize.
            base_optimizer: The class of the base optimizer (e.g. torch.optim.AdamW).
            rho (float): The neighborhood size (perturbation magnitude).
            adaptive (bool): Whether to use Adaptive SAM (ASAM).
            **kwargs: Keyword arguments passed to the base optimizer (lr, weight_decay, etc.).
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer
        # We pass self.param_groups to share the parameter groups reference
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Ensure param_groups are synchronized
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM:
        1. Computes the gradient norm.
        2. Calculates the perturbation epsilon.
        3. Adds epsilon to the weights (climb to the local maximum).
        4. Optionally zeros the gradients to prepare for the second backward pass.

        Args:
            zero_grad (bool): If True, clears gradients after the step.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save the original parameters to restore later
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation e_w
                # adaptive: scale by weight magnitude
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
        1. Restores the original weights.
        2. Updates the weights using the gradients computed at the perturbed state (via base_optimizer).
        3. Optionally zeros the gradients.

        Args:
            zero_grad (bool): If True, clears gradients after the step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original parameters: w
                p.data = self.state[p]["old_p"]

        # Update using the base optimizer: w = w - lr * grad(w_adv)
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using a closure.
        This allows SAM to be used with the standard optimizer.step() interface,
        but requires the closure to perform a full forward-backward pass.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        assert (
            closure is not None
        ), "SAM requires closure, but it is not supported without it."
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        Computes the global L2 norm of the gradients.
        Handles multi-device scenarios by projecting to a shared device.
        """
        # Identify a device to perform the norm calculation (usually the device of the first param)
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

    def zero_grad(self, set_to_none: bool = False):
        """
        Zeroes the gradients of all optimized parameters.
        Delegates to the base optimizer.
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        Delegates to the base optimizer to ensure compatibility.
        """
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        Delegates to the base optimizer and resyncs param_groups.
        """
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
