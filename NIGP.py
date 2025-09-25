# NIGP: Gaussian Process with Noisy Inputs (practical alternating implementation)
# Requires: numpy, scipy
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import GPy

# ---------------------------
# Kernel: Squared Exponential (ARD)
# ---------------------------
def SE_ARD_kernel(X1, X2, lengthscales, sigma_f):
    """
    X1: (n1, D)
    X2: (n2, D)
    lengthscales: (D,) positive
    sigma_f: scalar
    """
    kern=GPy.kern.RBF(input_dim=X1.shape[1],variance=sigma_f,lengthscale=lengthscales, ARD=True,inv_l=True)
    K=kern.K(X1,X2)
    return K

# ---------------------------
# Compute posterior mean gradients at arbitrary X_* (analytic)
# For SE kernel, derivative wrt x_* is:
#   d/dx_* k(x_*, x_j) = k(x_*, x_j) * (-(x_* - x_j) / lengthscale^2)
# So gradient of posterior mean at x_* = sum_j alpha_j * d/dx_* k(x_*, x_j)
# where alpha = (K + Sigma_noise)^-1 y
# ---------------------------
def compute_post_mean_and_gradients(X_train, y, lengthscales, sigma_f, sigma_y, noise_diag=None):
    """
    Returns:
      f_mean_train: (N,) posterior mean at training points computed with (K + Noise) used in inversion
      grads: (N, D) gradients of posterior mean wrt each input dim evaluated at each training point
    noise_diag: (N,) additional observation variance per point (heteroscedastic), excluding sigma_y^2
    """
    N, D = X_train.shape
    if noise_diag is None:
        noise_diag = np.zeros(N)

    K = SE_ARD_kernel(X_train, X_train, lengthscales, sigma_f)
    obs_noise = sigma_y**2 + noise_diag  # per-point variance
    K_noise = K + np.diag(obs_noise)
    cho = cho_factor(K_noise, lower=True)
    alpha = cho_solve(cho, y)   # (N,)

    # posterior mean at training inputs:
    f_mean = K @ alpha  # shape (N,)

    # derivatives:
    # For each i (test point x_* = X_train[i]), gradient = sum_j alpha_j * dk/dx_*(x_*, x_j)
    # where dk/dx_* = k(x_*, x_j) * (-(x_* - x_j) / l^2)
    grads = np.zeros((N, D))
    # Precompute K_ij
    Kmat = K  # shape (N,N)
    for i in range(N):
        # (x_i - x_j) along j
        diffs = (X_train[i:i+1, :] - X_train)  # shape (N, D)
        # factor: -(1 / l^2) broadcast across dims
        inv_ls2 = 1.0 / (lengthscales**2)
        # derivative per j: Kmat[i, j] * (-(x_i - x_j) * inv_ls2)
        # sum_j alpha_j * derivative_j
        # grads[i, d] = sum_j alpha_j * Kmat[i, j] * (-(x_i[d] - x_j[d]) * inv_ls2[d])
        weighted = (Kmat[i, :, None] * alpha[:, None]) * (-(diffs) * inv_ls2[None, :])
        grads[i, :] = np.sum(weighted, axis=0)
    return f_mean, grads

def compute_post_mean_and_gradients_fd(X_train, y, lengthscales, sigma_f, sigma_y, noise_diag=None, fd_eps=1e-3):
    """
    Compute posterior mean and its gradient wrt inputs using finite differences.

    Args:
      X_train: (N, D) training inputs
      y: (N,) targets
      lengthscales: (D,) ARD lengthscales
      sigma_f: scalar signal variance
      sigma_y: scalar noise std
      noise_diag: (N,) optional heteroscedastic extra variance
      fd_eps: finite-difference step size

    Returns:
      f_mean_train: (N,) posterior mean at training points
      grads: (N, D) finite-difference approximations to gradient of mean wrt each dim
    """
    N, D = X_train.shape
    if noise_diag is None:
        noise_diag = np.zeros(N)

    # build kernel matrix
    K = SE_ARD_kernel(X_train, X_train, lengthscales, sigma_f)
    obs_noise = sigma_y**2 + noise_diag  # per-point variance
    K_noise = K + np.diag(obs_noise)

    # solve (K+noise)^{-1} y
    cho = cho_factor(K_noise, lower=True)
    alpha = cho_solve(cho, y)   # (N,)

    # posterior mean at training inputs
    f_mean = K @ alpha  # (N,)

    # finite-difference gradients
    grads = np.zeros((N, D))
    for i in range(N):
        x0 = X_train[i:i+1, :]  # (1,D)
        for d in range(D):
            eps = fd_eps * max(1.0, abs(x0[0, d]))  # scale step to input magnitude
            x_plus = x0.copy(); x_plus[0, d] += eps
            x_minus = x0.copy(); x_minus[0, d] -= eps

            # compute mean at perturbed points
            k_plus = SE_ARD_kernel(x_plus, X_train, lengthscales, sigma_f)  # (1,N)
            k_minus = SE_ARD_kernel(x_minus, X_train, lengthscales, sigma_f)
            f_plus = float(k_plus @ alpha)
            f_minus = float(k_minus @ alpha)

            grads[i, d] = (f_plus - f_minus) / (2 * eps)

    return f_mean, grads
 
def safe_obj(lh, X, y, grad_fixed, noise_diag_extra_fixed):
    val = neg_log_marginal_likelihood(lh, X, y, grad_fixed, noise_diag_extra_fixed)
    if not np.isfinite(val):
        return 1e20
    return val   
# ---------------------------
# Marginal likelihood (with fixed gradients) to optimize hyperparameters:
# We parameterize hyperparams in log-space for positivity.
# hyperparams vector: [log_lengthscales (D), log_sigma_f, log_sigma_y, log_sigma_x (D)]
# where sigma_x are per-dimension input-noise stddevs (not variances)
# ---------------------------
def neg_log_marginal_likelihood(log_hyp, X, y, grad_fixed, noise_diag_extra_fixed=None):
    """
    grad_fixed: (N, D) the gradients used to compute per-point extra variance from input noise.
      We treat grads fixed for this optimization step.
    noise_diag_extra_fixed: optional (N,) if we want to include any pre-existing per-point noise addition.
    """
    N, D = X.shape
    # parse params
    ls = np.exp(log_hyp[:D])            # lengthscales
    sigma_f = np.exp(log_hyp[D])        # signal std
    sigma_y = np.exp(log_hyp[D+1])      # output noise std
    sigma_x = np.exp(log_hyp[D+2:])     # input-noise std per dimension

    # per-point extra noise from input noise: v_i = sum_d (grad_i_d^2 * sigma_x_d^2)
    v = np.sum((grad_fixed**2) * (sigma_x[None, :]**2), axis=1)  # shape (N,)
    if noise_diag_extra_fixed is not None:
        v = v + noise_diag_extra_fixed

    # build K and compute log marginal likelihood
    K = SE_ARD_kernel(X, X, ls, sigma_f)
    obs_var = sigma_y**2 + v
    K_noise = K + np.diag(obs_var)
    jitter = 1e-8
    try:
        cho = cho_factor(K_noise+ np.eye(N)*jitter, lower=True)
        alpha = cho_solve(cho, y)
    except np.linalg.LinAlgError:
        # ill-conditioned -> large penalty
        return 1e25
    logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
    #logdet2=np.log(np.linalg.det(K_noise))
    #print(logdet,logdet2)
    nlml = 0.5 * y.T @ alpha + 0.5 * logdet + 0.5 * N * np.log(2*np.pi)
    # faster: use cho factor to compute logdet
    # but for simplicity / clarity we used np.linalg.det above (N small usually)
    return float(nlml)

# ---------------------------
# NIGP class
# ---------------------------
class NIGP:
    def __init__(self, n_restarts=3, iters=3, verbose=True):
        """
        n_restarts: number of random restarts for hyperparameter optimization
        iters: number of alternations between gradient computation and hyperparameter optimization
        """
        self.n_restarts = n_restarts
        self.iters = iters
        self.verbose = verbose
        # placeholders for learned hyperparams
        self.lengthscales_ = None
        self.sigma_f_ = None
        self.sigma_y_ = None
        self.sigma_x_ = None  # per-dim input noise std
        self.X_train_ = None
        self.y_train_ = None
        self.noise_diag_train_ = None  # per-point extra noise (v_i)
   
    def get_params(self):
        return np.hstack((self.sigma_x_,self.sigma_f_,self.sigma_y_,self.lengthscales_))
    
    def fit(self, X, y, maxiter_opt=200):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        N, D = X.shape
        self.X_train_ = X
        self.y_train_ = y

        # initial hyperparameters (log-space)
        # lengthscales: median pairwise dist or 1.0
        pairwise = np.sqrt(np.maximum(0, np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)))
        med = np.median(pairwise[pairwise>0]) if np.any(pairwise>0) else 1.0
        init_ls = np.ones(D) * (med if med>0 else 1.0)
        init_sigma_f = np.std(y) if np.std(y)>0 else 1.0
        init_sigma_y = 0.1 * init_sigma_f
        init_sigma_x = np.ones(D) * 0.01 * np.std(X, axis=0)  # small initial input noise

        # initialize gradients zero (first optimize ignoring input noise)
        grad_fixed = np.zeros((N, D))
        noise_diag_extra_fixed = np.zeros(N)

        # initialize hyperparams (in log)
        log_hyp = np.concatenate([np.log(init_ls), [np.log(init_sigma_f), np.log(init_sigma_y)], np.log(np.maximum(init_sigma_x, 1e-8))])

        # alternate
        for it in range(self.iters):
            if self.verbose:
                print(f"NIGP iteration {it+1}/{self.iters} ...")
            # Step A: compute posterior mean gradients using current hyperparams (but treat input-noise effect in obs as zero initially)
            ls = np.exp(log_hyp[:D])
            sigma_f = np.exp(log_hyp[D])
            sigma_y = np.exp(log_hyp[D+1])
            f_mean_train, grads = compute_post_mean_and_gradients(X, y, ls, sigma_f, sigma_y, noise_diag=None)
            #f_mean_train, grads = compute_post_mean_and_gradients_fd(X, y, ls, sigma_f, sigma_y, noise_diag=None)
            # use grads as fixed for optimization
            grad_fixed = grads

            # Step B: optimize hyperparameters given these grads
            best = None
            best_val = 1e99
            # perform n_restarts restarts
            for restart in range(self.n_restarts):
                # randomize starting point slightly
                init = log_hyp + 0.1 * np.random.randn(*log_hyp.shape)
                bounds = [(np.log(1e-6), np.log(1e6))]* (D + 2 + D)  # broad bounds for all
                res = minimize(lambda lh: safe_obj(lh, X, y, grad_fixed, noise_diag_extra_fixed),
                               init, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter_opt, 'disp': False})
                if res.fun < best_val:
                    best_val = res.fun
                    best = res
            log_hyp = res.x if best is None else best.x
            if self.verbose:
                print(f"  optimized nlml: {best_val:.6g}")

        # save learned hyperparams
        D = X.shape[1]
        self.lengthscales_ = np.exp(log_hyp[:D])
        self.sigma_f_ = np.exp(log_hyp[D])
        self.sigma_y_ = np.exp(log_hyp[D+1])
        self.sigma_x_ = np.exp(log_hyp[D+2:])
        # compute final per-point extra noise
        v = np.sum((grad_fixed**2) * (self.sigma_x_[None, :]**2), axis=1)
        self.noise_diag_train_ = v

        if self.verbose:
            print("Learned hyperparameters:")
            print(" lengthscales:", self.lengthscales_)
            print(" sigma_f:", self.sigma_f_)
            print(" sigma_y:", self.sigma_y_)
            print(" sigma_x (per-dim):", self.sigma_x_)
        return self

    # ---------------------------
    # Predict
    # Use eq (7) style: treat training points deterministic and add diag{Delta fbar Sigma_x Delta fbar^T}
    # Predictive mean: k(x*,X) @ (K + diag(obs_var))^{-1} y
    # Predictive variance: k(x*,x*) - k(x*,X) (K + diag(obs_var))^{-1} k(X,x*)
    # (Optionally add predictive input-noise at x* if provided)
    # ---------------------------
    def predict(self, Xs, Xs_input_noise=None, return_var=True, return_cov=False):
        """
        Predictive mean, variance, or covariance from GP.
    
        Args:
            Xs : (M,D) test inputs
            Xs_input_noise : optional, (M,D) input noise std per dim, or (D,), or scalar
            return_var : if True, return marginal variances (default)
            return_cov : if True, return full covariance matrix
    
        Returns:
            mean : (M,)
            var  : (M,) marginal variances if return_var=True
            cov  : (M,M) full covariance matrix if return_cov=True
        """
        Xs = np.asarray(Xs, dtype=float)
        K = SE_ARD_kernel(self.X_train_, self.X_train_, self.lengthscales_, self.sigma_f_)
        obs_var = self.sigma_y_**2 + (self.noise_diag_train_ if self.noise_diag_train_ is not None else 0.0)
        K_noise = K + np.diag(obs_var)
        cho = cho_factor(K_noise, lower=True)
        alpha = cho_solve(cho, self.y_train_)
    
        # cross-covariance
        Kxs = SE_ARD_kernel(Xs, self.X_train_, self.lengthscales_, self.sigma_f_)  # (M, N)
        mean = Kxs @ alpha  # (M,)
    
        if not (return_var or return_cov):
            return mean
    
        # predictive covariance
        Ksxs = SE_ARD_kernel(Xs, Xs, self.lengthscales_, self.sigma_f_)  # (M,M)
        v = cho_solve(cho, Kxs.T)  # (N, M)
        cov = Ksxs - Kxs @ v       # (M,M)
    
        # if input noise correction requested
        if Xs_input_noise is not None:
            M, D = Xs.shape
            grads_star = np.zeros((M, D))
            for m in range(M):
                diffs = (Xs[m:m+1, :] - self.X_train_)  # (N,D)
                inv_ls2 = 1.0 / (self.lengthscales_**2)
                weighted = (Kxs[m, :, None] * alpha[:, None]) * (-(diffs) * inv_ls2[None, :])
                grads_star[m, :] = np.sum(weighted, axis=0)
    
            Xs_input_noise = np.asarray(Xs_input_noise)
            if Xs_input_noise.ndim == 1 and Xs_input_noise.size == D:
                Sigma_x_star = Xs_input_noise[None, :]
            elif Xs_input_noise.shape == grads_star.shape:
                Sigma_x_star = Xs_input_noise
            else:
                raise ValueError("Xs_input_noise must be scalar, shape (D,) or (M,D)")
    
            v_star = np.sum((grads_star**2) * (Sigma_x_star**2), axis=1)
    
            # add to diagonal of covariance
            cov = cov + np.diag(v_star)
    
        # numerical floor for stability
        cov = cov + np.eye(cov.shape[0]) * 1e-12
    
        if return_cov:
            return mean, cov
        else:
            var = np.maximum(np.diag(cov), 1e-12)
            return mean, var


# ---------------------------
# Example usage with synthetic data
# ---------------------------
if __name__ == "__main__":
    np.random.seed(0)
    # 1D toy data y = sin(x), but inputs have measurement noise
    N = 40
    X_true = np.linspace(0, 6, N)[:, None]
    y_true = np.sin(X_true).ravel()
    
    #X_true = np.linspace(0, 60, N)[:, None]
    #y_true = np.sin(X_true/5).ravel()
    # add output noise and input noise
    sigma_y_true = 0.05
    sigma_x_true = .2  # true input noise (std)
    X_obs = X_true + sigma_x_true * np.random.randn(*X_true.shape)
    y_obs = y_true + sigma_y_true * np.random.randn(N)

    model = NIGP(n_restarts=2, iters=10, verbose=True)
    model.fit(X_obs, y_obs)

    Xtest = np.linspace(-0.5, 6.5, 200)[:, None]
    mean, var = model.predict(Xtest, Xs_input_noise=np.ones(Xtest.shape) * model.sigma_x_)
    import matplotlib.pyplot as plt
    plt.fill_between(Xtest.ravel(), mean - 3*np.sqrt(var), mean + 3*np.sqrt(var), alpha=0.3)
    plt.plot(Xtest, mean, label='NIGP mean')
    plt.scatter(X_obs.ravel(), y_obs, c='k', s=20, label='noisy observations')
    plt.scatter(X_true.ravel(), y_obs, c='r', s=20,marker='x', label='true observations')
    plt.legend()
    plt.show()
