import numpy as np
from scipy.optimize import minimize
from library import config


def _one_hot_encode(y, n_classes):
    """
    Converts integer labels to one-hot encoded matrix in float64.

    Args:
        y (np.ndarray): Integer labels of shape (N,).
        n_classes (int): Number of classes K.

    Returns:
        np.ndarray: One-hot matrix of shape (N, K).
    """
    N = y.shape[0]
    Y = np.zeros((N, n_classes), dtype=config.FLOAT_PRECISION)
    Y[np.arange(N), y] = 1.0
    return Y


def _calibration_objective(params, Z, Y_true, reg_strength, b_init):
    """
    Computes the regularized Log Loss and its gradient w.r.t scale (s) and bias (b).

    Objective:
        J(s, b) = -1/N * sum(Y * log(Softmax(s*Z + b))) + (lambda/2) * ||b - b_init||^2

    Args:
        params (np.ndarray): Flattened array [s, b_0, b_1, ..., b_K-1].
        Z (np.ndarray): Generative logits (X @ W.T) of shape (N, K).
        Y_true (np.ndarray): One-hot targets of shape (N, K).
        reg_strength (float): L2 regularization coefficient.
        b_init (np.ndarray): Initial analytical bias vector of shape (K,).

    Returns:
        tuple: (loss (float), gradient (np.ndarray))
    """
    # 1. Unpack parameters
    s = params[0]
    b = params[1:]  # Shape (K,)

    N, K = Z.shape

    # 2. Compute Calibrated Logits: L = s * Z + b
    # Broadcasting b: (N, K) + (K,) -> (N, K)
    logits = s * Z + b

    # 3. Stable Softmax / Log-Softmax
    # Subtract max for numerical stability to prevent overflow in exp()
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - max_logits
    exp_logits = np.exp(shifted_logits)
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    log_sum_exp = np.log(sum_exp)

    # log_probs = logits - log(sum(exp(logits)))
    #           = (shifted + max) - (log(sum_exp_shifted) + max)
    #           = shifted - log(sum_exp_shifted)
    log_probs = shifted_logits - log_sum_exp

    # 4. Compute Log Loss (Negative Log Likelihood)
    # Sum over all classes and samples: - sum(Y_ik * log(P_ik)) / N
    nll = -np.sum(Y_true * log_probs) / N

    # 5. Compute Regularization: 0.5 * lambda * ||b - b_init||^2
    # We only regularize the bias drift, not the scale.
    b_diff = b - b_init
    reg_loss = 0.5 * reg_strength * np.sum(b_diff**2)

    total_loss = nll + reg_loss

    # 6. Compute Gradients
    # Probabilities P_ik
    probs = exp_logits / sum_exp

    # Gradient of NLL w.r.t Logits L_ik: (P_ik - Y_ik) / N
    d_logits = (probs - Y_true) / N

    # Gradient w.r.t scale s:
    # dJ/ds = sum_{i,k} (dJ/dL_ik * dL_ik/ds)
    # dL_ik/ds = Z_ik
    # dJ/ds = sum(d_logits * Z)
    d_s = np.sum(d_logits * Z)

    # Gradient w.r.t bias b_k:
    # dJ/db_k = sum_{i} (dJ/dL_ik) + d(Reg)/db_k
    # d(Reg)/db_k = lambda * (b_k - b_init_k)
    d_b = np.sum(d_logits, axis=0) + reg_strength * b_diff

    # Flatten gradient
    grad = np.concatenate(([d_s], d_b))

    return total_loss, grad


def optimize_calibration(Z, y, b_init):
    """
    Optimizes the scale scalar and bias vector to minimize Log Loss.

    Args:
        Z (np.ndarray): Generative logits (N, K).
        y (np.ndarray): Integer target labels (N,).
        b_init (np.ndarray): Initial bias vector (K,) derived from generative priors.

    Returns:
        tuple: (s_opt (float), b_opt (np.ndarray))
    """
    # Ensure inputs are float64
    Z = Z.astype(config.FLOAT_PRECISION)
    b_init = b_init.astype(config.FLOAT_PRECISION)

    N, K = Z.shape

    # One-hot encode targets
    Y_true = _one_hot_encode(y, K)

    # Initial parameters: s=1.0, b=b_init
    # We start with the generative solution which is a strong prior
    s_init = 1.0
    initial_params = np.concatenate(([s_init], b_init))

    # Optimization Configuration
    opt_config = config.CONFIG["optimization"]
    reg_strength = config.CONFIG["calibration_reg"]

    print(f"Starting L-BFGS-B optimization for calibration (N={N}, K={K})...")

    # Run Optimization
    result = minimize(
        fun=_calibration_objective,
        x0=initial_params,
        args=(Z, Y_true, reg_strength, b_init),
        method="L-BFGS-B",
        jac=True,  # We provide the analytical gradient
        options={
            "maxiter": opt_config["maxiter"],
            "ftol": opt_config["ftol"],
            "gtol": opt_config["gtol"],
            "disp": False,
        },
    )

    if not result.success:
        print(f"Warning: Calibration optimization did not converge: {result.message}")

    # Extract results
    s_opt = result.x[0]
    b_opt = result.x[1:]

    print(f"Calibration complete. Final Loss: {result.fun:.6f}")
    print(f"Optimized Scale: {s_opt:.4f}")
    print(f"Bias Drift (L2): {np.linalg.norm(b_opt - b_init):.4f}")

    return s_opt, b_opt
