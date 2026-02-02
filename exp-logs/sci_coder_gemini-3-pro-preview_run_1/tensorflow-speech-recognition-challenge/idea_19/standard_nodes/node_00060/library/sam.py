import torch
import torch.nn as nn
import torch.optim as optim


class SAM(torch.optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) optimizer.

    Wraps a base optimizer (e.g., SGD, AdamW) and performs a two-step update
    to minimize both loss value and loss sharpness (landscape flatness).

    This implementation requires the training loop to explicitly call
    first_step() and second_step().

    References:
        Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: The class of the base optimizer (e.g. torch.optim.AdamW).
            rho: The neighborhood size for the ascent step (default: 0.05).
            adaptive: Whether to use adaptive SAM (ASAM) (default: False).
            **kwargs: Keyword arguments passed to the base optimizer.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer
        # We pass self.param_groups so that parameter groups are shared/synced
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Ensure param_groups are synchronized
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the ascent step:
        1. Computes gradient norm.
        2. Calculates perturbation epsilon.
        3. Saves current weights.
        4. Applies perturbation to weights.

        Args:
            zero_grad (bool): If True, clears gradients after the step.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save current parameters to state to restore later
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation e_w
                # Adaptive SAM scales by |w|^2, Standard SAM uses 1.0
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
        Performs the descent step:
        1. Restores original weights.
        2. Updates weights using the gradients computed at the perturbed state.

        Args:
            zero_grad (bool): If True, clears gradients after the step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameters
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Update using the base optimizer (using gradients from w_adv)
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        Standard step method is not supported directly because SAM requires
        two forward/backward passes with intervention in between.
        Use first_step() and second_step() explicitly in the training loop.
        """
        raise NotImplementedError(
            "SAM requires step to be split into first_step and second_step. "
            "Please update your training loop to call these explicitly."
        )

    def _grad_norm(self):
        """
        Calculates the global gradient norm across all parameter groups.
        """
        # Find a reference device to stack tensors
        shared_device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    shared_device = p.device
                    break
            if shared_device is not None:
                break

        if shared_device is None:
            return 0.0

        # Compute norm of all gradients concatenated
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
        Loads the state of the base optimizer.
        """
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
