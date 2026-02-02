import torch
import torch.optim as optim


class SAM(optim.Optimizer):
    """
    SAM (Sharpness-Aware Minimization) Optimizer.

    Implements the SAM algorithm which minimizes both the loss value and the loss sharpness,
    thereby seeking parameters that lie in neighborhoods having uniformly low loss.

    References:
        - "Sharpness-Aware Minimization for Efficiently Improving Generalization" (Foret et al., 2021)
        - "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning" (Kwon et al., 2021)
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: Class of the base optimizer (e.g. torch.optim.AdamW).
            rho: Neighborhood size (default: 0.05).
            adaptive: If True, implements Adaptive SAM (ASAM).
            **kwargs: Keyword arguments passed to the base_optimizer.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize the base optimizer with the parameter groups managed by this class
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

        # Sync param_groups and defaults
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the ascent step: computes perturbation and applies it to parameters.

        Args:
            zero_grad (bool): If True, zeros the gradients after applying perturbation.
                              Required if the closure accumulates gradients.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            # Calculate scale factor: rho / (norm + epsilon)
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Save current parameter state to restore later
                self.state[p]["old_p"] = p.data.clone()

                # Calculate perturbation e_w
                # Standard SAM: e_w = rho * grad / norm
                # Adaptive SAM: e_w = rho * |w|^2 * grad / norm_adaptive
                if group["adaptive"]:
                    e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)

                # Apply perturbation: w_adv = w + e_w
                p.add_(e_w)

                if zero_grad:
                    p.grad.zero_()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the descent step: restores parameters and updates using base optimizer.

        Args:
            zero_grad (bool): If True, zeros the gradients after the update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original parameters: w = w_adv - e_w
                # We use the saved state directly to avoid numerical drift
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Update parameters using the gradients computed at w_adv
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
        assert closure is not None, "SAM requires closure, but it was not provided."

        # Wrap closure to ensure gradients are enabled during re-evaluation
        closure = torch.enable_grad()(closure)

        # 1. Ascent Step: Perturb weights
        self.first_step(zero_grad=True)

        # 2. Re-evaluate loss and gradients at perturbed state
        loss = closure()

        # 3. Descent Step: Restore weights and update using base optimizer
        self.second_step()

        return loss

    def _grad_norm(self):
        """
        Computes the global norm of gradients across all parameter groups.
        Handles device placement correctly.
        """
        # Collect norms from all parameters
        norm_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # For ASAM, the norm is weighted by parameter magnitude
                    if group["adaptive"]:
                        element = torch.abs(p) * p.grad
                    else:
                        element = p.grad

                    norm_list.append(element.norm(p=2))

        if not norm_list:
            return torch.tensor(0.0)

        # Ensure all norms are on the same device before stacking
        # We use the device of the first norm found
        device = norm_list[0].device
        norm_list = [n.to(device) for n in norm_list]

        # Compute global norm
        total_norm = torch.norm(torch.stack(norm_list), p=2)
        return total_norm
