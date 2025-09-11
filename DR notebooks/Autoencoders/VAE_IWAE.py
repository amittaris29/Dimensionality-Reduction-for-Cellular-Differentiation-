import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import functools as ft
import jax.nn as jnn
from typing import Any
from typing import List
from typing import Sequence, Optional, List, Callable, Tuple


class DeepVAE(eqx.Module):
    """
    Deep Variational Autoencoder (Equinox + JAX).

    Architecture
    -----------
    - Encoder: MLP -> produces μ and logσ²
    - Reparameterization: z = μ + σ ⊙ ε
    - Decoder: MLP -> reconstructs x̂

    Notes
    -----
    The encoder/decoder widths are configurable via `encoder_hidden` and
    `decoder_hidden`. If `decoder_hidden=None`, the decoder mirrors the
    encoder widths in reverse order.
    """

    # Learned submodules
    encoder_layers: List[eqx.nn.Linear]
    mu_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear
    decoder_layers: List[eqx.nn.Linear]

    # Saved hyperparameters
    latent_dim: int
    input_dim: int
    encoder_hidden: Sequence[int]
    decoder_hidden: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray]

    def __init__(
        self,
        key: jax.random.PRNGKey,
        *,
        latent_dim: int = 2,
        input_dim: int = 3,
        encoder_hidden: Sequence[int] = (32, 16, 8),
        decoder_hidden: Optional[Sequence[int]] = None,  # if None, mirror encoder
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jnn.relu,
    ):
        """
        Initialize the VAE.

        Args:
            key: PRNG key for parameter initialization.
            latent_dim: Dimension of latent space z.
            input_dim: Dimension of input vector x (flattened).
            encoder_hidden: Sequence of hidden widths for encoder MLP.
            decoder_hidden: Sequence of hidden widths for decoder MLP. If None,
                            uses reversed encoder_hidden.
            activation: Nonlinearity used between Linear layers.
        """
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder_hidden = tuple(encoder_hidden)
        self.decoder_hidden = (
            tuple(decoder_hidden) if decoder_hidden is not None
            else tuple(reversed(encoder_hidden))
        )
        self.activation = activation

        k_enc, k_mu, k_logvar, k_dec = jax.random.split(key, 4)

        # ---- Build encoder ----
        enc_sizes = [input_dim, *self.encoder_hidden]
        self.encoder_layers = [
            eqx.nn.Linear(enc_sizes[i], enc_sizes[i + 1], key=jax.random.fold_in(k_enc, i))
            for i in range(len(enc_sizes) - 1)
        ]
        last_enc_width = enc_sizes[-1]
        self.mu_layer = eqx.nn.Linear(last_enc_width, latent_dim, key=k_mu)
        self.logvar_layer = eqx.nn.Linear(last_enc_width, latent_dim, key=k_logvar)

        # ---- Build decoder ----
        dec_sizes = [latent_dim, *self.decoder_hidden, input_dim]
        self.decoder_layers = [
            eqx.nn.Linear(dec_sizes[i], dec_sizes[i + 1], key=jax.random.fold_in(k_dec, i))
            for i in range(len(dec_sizes) - 1)
        ]

    # ------------------------------------------------------------
    # Mini-function explainer: Encoder
    # Maps x -> (mu, logvar) for diagonal Gaussian q(z|x)
    # ------------------------------------------------------------
    def encode(self, x: Float[Array, "input_dim"]) -> Tuple[
        Float[Array, "latent_dim"], Float[Array, "latent_dim"]
    ]:
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    # ------------------------------------------------------------
    # Mini-function explainer: Reparameterization trick
    # Draws z ~ N(mu, diag(exp(logvar))) in a differentiable way.
    # ------------------------------------------------------------
    def reparameterize(
        self,
        mu: Float[Array, "latent_dim"],
        logvar: Float[Array, "latent_dim"],
        key: jax.random.PRNGKey,
    ) -> Float[Array, "latent_dim"]:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, shape=std.shape)
        return mu + eps * std

    # ------------------------------------------------------------
    # Mini-function explainer: Decoder
    # Maps latent z -> reconstruction x_hat
    # ------------------------------------------------------------
    def decode(self, z: Float[Array, "latent_dim"]) -> Float[Array, "input_dim"]:
        for layer in self.decoder_layers[:-1]:
            z = self.activation(layer(z))
        return self.decoder_layers[-1](z)

    # ------------------------------------------------------------
    # Mini-function explainer: Full forward pass
    # x -> (x_hat, mu, logvar), with stochastic z via reparameterize
    # ------------------------------------------------------------
    def __call__(
        self,
        x: Float[Array, "input_dim"],
        key: jax.random.PRNGKey,
    ) -> Tuple[
        Float[Array, "input_dim"],
        Float[Array, "latent_dim"],
        Float[Array, "latent_dim"],
    ]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, key)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# ============================================================
#                  VAE/IWAE Loss Functions
# ============================================================


# ===========================
# Riemannian / geometry helpers
# ===========================

def _metric_scale(likelihood: str, sigma_x: float) -> float:
    # Fisher metric scale for Gaussian; unity for Bernoulli/logits proxy
    return (1.0 / (sigma_x ** 2)) if likelihood == "gaussian" else 1.0


def quadform_G(model, z, v, *, likelihood="gaussian", sigma_x=1.0):
    """
    v^T G(z) v = || J_g(z) v ||^2  (times 1/sigma_x^2 for Gaussian)
    Uses a single JVP; very cheap.
    """
    _, Jv = jax.jvp(model.decode, (z,), (v,))
    return _metric_scale(likelihood, sigma_x) * jnp.sum(Jv * Jv)


def trace_G(model, z, *, likelihood="gaussian", sigma_x=1.0, num_probe: int = 2, key=None):
    """
    Hutchinson estimator for tr(G(z)) = ||J_g(z)||_F^2.
    num_probe in {1..4} is typically enough.
    """
    key = key or jax.random.PRNGKey(0)
    scale = _metric_scale(likelihood, sigma_x)

    def single(v):
        _, Jv = jax.jvp(model.decode, (z,), (v,))
        return jnp.sum(Jv * Jv)

    # Random unit probes in R^d
    vs = jax.random.normal(key, shape=(num_probe, z.shape[-1]))
    vs = vs / (jnp.linalg.norm(vs, axis=1, keepdims=True) + 1e-12)
    vals = jax.vmap(single)(vs)
    return scale * jnp.mean(vals)


def gram_G(model, z, *, likelihood="gaussian", sigma_x=1.0):
    """
    Build the small d×d Gram G(z) = J^T J (scaled) and its logdet in a stable way.
    Cost: d JVPs, fine when latent_dim d is small (e.g., <= 16).
    """
    d = z.shape[-1]
    eye_d = jnp.eye(d)

    # Columns of J via JVP with basis vectors e_i
    def col(ei):
        _, Jvi = jax.jvp(model.decode, (z,), (ei,))
        return Jvi  # shape (D,)

    J_cols = jax.vmap(col)(eye_d)       # (d, D)
    J = jnp.transpose(J_cols, (1, 0))   # (D, d)

    G = J.T @ J                         # (d, d)
    G = G * _metric_scale(likelihood, sigma_x)

    # Stabilize and get logdet
    epsI = 1e-8 * jnp.eye(d)
    sign, logdet = jnp.linalg.slogdet(G + epsI)
    logdet = jnp.where(sign > 0, logdet, -jnp.inf)
    return G, logdet

# Log-probability helpers
# -----------------------------
def _log_normal_diag(x, mu, logvar):  # returns per-sample log N(x; mu, diag(exp(logvar)))
    # sum over last dim
    return -0.5 * (jnp.sum(jnp.log(2 * jnp.pi) + logvar + (x - mu) ** 2 / jnp.exp(logvar), axis=-1))

def _log_standard_normal(z):          # log N(z; 0, I), sum over last dim
    return -0.5 * (jnp.sum(jnp.log(2 * jnp.pi) + z**2, axis=-1))

def _bernoulli_loglik_with_logits(x, logits):  # sum over last dim
    # x in [0,1], logits are decoder outputs (no sigmoid needed)
    return jnp.sum(x * jax.nn.log_sigmoid(logits) + (1.0 - x) * jax.nn.log_sigmoid(-logits), axis=-1)

def _gaussian_loglik(x, mean, sigma): # factorized Gaussian with fixed sigma>0; sum over last dim
    var = sigma**2
    return -0.5 * (jnp.sum((x - mean)**2 / var + jnp.log(2 * jnp.pi * var), axis=-1))

# ---------------------------------------------------------
# Unified loss: ELBO (default) or IWAE when iwae=True
# Keep your trainer the same by using functools.partial.
# ---------------------------------------------------------
def loss2_VAE(
    params, static, x, key, *,
    iwae: bool = False,
    K: int = 5,
    likelihood: str = "gaussian",
    sigma_x: float = 0.7071,
    beta: float = 0.7,
    alpha: float = 1e-5,
    # ---- new: geometry knobs (all optional; default OFF) ----
    lambda_mf: float = 0.0,         # weight for MF uniformity (var(logdet G))
    lambda_iso: float = 0.0,        # weight for local isometry penalty
    geo_where: str = "posterior",   # "posterior" or "prior"
    geo_at: str = "mu",             # "mu" or "sample" (posterior only)
    geo_detach_z: bool = False      # True => stop-gradient on z for geometry terms
):
    """
    Unified VAE loss with optional Riemannian geometry regularizers.

    Geometry terms (averaged over batch):
      - MF uniformity: var(logdet G(z))
      - Isometry: || G(z)/(tr(G)/d) - I ||_F^2
    Both use G(z) = J_g(z)^T J_g(z) (scaled by 1/sigma_x^2 for Gaussian).
    """

    model = eqx.combine(params, static)
    B, D = x.shape

    # ----------------- Base objective (ELBO or IWAE) -----------------
    mu, logvar = jax.vmap(model.encode)(x)           # (B, L)
    std = jnp.exp(0.5 * logvar)

    if not iwae:
        # ELBO (one-sample MC on recon term)
        eps = jax.random.normal(key, shape=mu.shape)
        z_samp = mu + std * eps                       # (B, L)
        xhat = jax.vmap(model.decode)(z_samp)         # (B, D)

        if likelihood == "bernoulli_logits":
            recon_ll = _bernoulli_loglik_with_logits(x, xhat)  # (B,)
            recon_nll = -jnp.mean(recon_ll)
        elif likelihood == "gaussian":
            recon_ll = _gaussian_loglik(x, xhat, sigma_x)      # (B,)
            recon_nll = -jnp.mean(recon_ll)
        else:
            raise ValueError("likelihood must be 'bernoulli_logits' or 'gaussian'.")

        kl = -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))

        # L2 on weights
        def _is_weight(p): return isinstance(p, jnp.ndarray) and p.ndim > 1
        l2 = sum(jnp.sum(p * p)
                 for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
                 if _is_weight(p))

        base_loss = recon_nll + beta * kl + alpha * l2

        # We'll reuse z_samp if geo_at="sample"
    else:
        # IWAE (K importance samples)
        eps = jax.random.normal(key, shape=(B, K, mu.shape[-1]))   # (B,K,L)
        zK = mu[:, None, :] + std[:, None, :] * eps                # (B,K,L)

        # Decode
        xhat_flat = jax.vmap(model.decode)(zK.reshape(B * K, -1))  # (B*K, D)
        xhatK = xhat_flat.reshape(B, K, D)                         # (B,K,D)

        # Likelihoods
        x_exp = x[:, None, :]
        if likelihood == "bernoulli_logits":
            log_px = _bernoulli_loglik_with_logits(x_exp, xhatK)   # (B,K)
        elif likelihood == "gaussian":
            log_px = _gaussian_loglik(x_exp, xhatK, sigma_x)       # (B,K)
        else:
            raise ValueError("likelihood must be 'bernoulli_logits' or 'gaussian'.")

        # Prior & posterior
        log_pz = _log_standard_normal(zK)                          # (B,K)
        muK = jnp.repeat(mu[:, None, :], K, axis=1)
        logvarK = jnp.repeat(logvar[:, None, :], K, axis=1)
        log_qz = _log_normal_diag(zK, muK, logvarK)                # (B,K)

        # IWAE bound
        log_w = log_px + log_pz - log_qz                           # (B,K)
        log_w_norm = jax.scipy.special.logsumexp(log_w, axis=1) - jnp.log(K)  # (B,)
        iwae_bound = jnp.mean(log_w_norm)
        # Optional L2
        def _is_weight(p): return isinstance(p, jnp.ndarray) and p.ndim > 1
        l2 = sum(jnp.sum(p * p)
                 for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
                 if _is_weight(p))
        base_loss = (-iwae_bound) + alpha * l2

        # For geometry "sample" choice we will sample fresh below (simpler)

    # ----------------- Geometry regularizers (optional) -----------------
    if (lambda_mf > 0.0) or (lambda_iso > 0.0):
        # Choose where to evaluate geometry: z_geo ∈ R^{B×L}
        if geo_where == "prior":
            key, sub = jax.random.split(key)
            z_geo = jax.random.normal(sub, shape=mu.shape)
        else:  # "posterior"
            if geo_at == "mu":
                z_geo = mu
            elif geo_at == "sample":
                key, sub = jax.random.split(key)
                eps_geo = jax.random.normal(sub, shape=mu.shape)
                z_geo = mu + std * eps_geo
            else:
                raise ValueError("geo_at must be 'mu' or 'sample'.")

        if geo_detach_z:
            z_geo = jax.lax.stop_gradient(z_geo)

        # Compute per-sample G (small d×d) and its logdet
        def _G_and_logdet(z_i):
            return gram_G(model, z_i, likelihood=likelihood, sigma_x=sigma_x)
        Gs, logdets = jax.vmap(_G_and_logdet, in_axes=0)(z_geo)  # Gs: (B,d,d); logdets: (B,)

        # -- (a) MF uniformity: minimize var(logdet G)
        mf_pen = jnp.var(logdets)
        #print(mf_pen)
          
        # -- (b) Isometry (scale-invariant): || G/(tr(G)/d) - I ||_F^2
        d = mu.shape[-1]
        trG = jnp.trace(Gs, axis1=1, axis2=2) + 1e-12            # (B,)
        scale = (trG / d)[:, None, None]                          # (B,1,1)
        Gn = Gs / scale                                           # normalized
        iso_pen = jnp.mean(jnp.sum((Gn - jnp.eye(d)) ** 2, axis=(1, 2)))

        geo_loss = lambda_iso * iso_pen + lambda_mf #* mf_pen 
    else:
        geo_loss = 0.0

    # ----------------- Total -----------------
    return base_loss + geo_loss


# --- assumes the following helpers from your loss are already defined:
# _bernoulli_loglik_with_logits(x, logits)
# _gaussian_loglik(x, mean, sigma)
# loss2_VAE(...)  # your unified ELBO/IWAE loss

# -----------------------------
# Helpers for evaluation
# -----------------------------
def _as_jnp_batch(x_batch, input_dim):
    """Convert a batch (possibly torch tensor; possibly (B,1,H,W)) to jnp.float32 (B, D)."""
    if hasattr(x_batch, "numpy"):  # torch tensor
        x_batch = x_batch.numpy()
    x = jnp.array(x_batch).astype(jnp.float32)
    if x.ndim > 2:  # e.g., (B,1,28,28) -> (B, 28*28)
        x = x.reshape(x.shape[0], -1)
    assert x.shape[1] == input_dim, f"Expected input_dim={input_dim}, got {x.shape[1]}"
    return x

def reconstruct_deterministic(model, x_batch):
    """
    Deterministic reconstruction: decode from μ(x) (no sampling).
    Returns logits (Bernoulli) or mean (Gaussian), matching your decoder output.
    """
    mu, _ = jax.vmap(model.encode)(x_batch)     # (B, latent_dim)
    xhat = jax.vmap(model.decode)(mu)           # (B, D); logits if Bernoulli, mean if Gaussian
    return xhat

def compute_recon_metrics(
    model,
    x_batch,
    *,
    likelihood: str = "bernoulli_logits",
    sigma_x: float = 0.1
):
    """
    Returns a dict with:
      - recon_mse: mean per-sample MSE in data space ([0,1] for Bernoulli)
      - recon_nll: negative log-likelihood (lower is better)
    """
    xhat = reconstruct_deterministic(model, x_batch)  # (B, D)

    if likelihood == "bernoulli_logits":
        # For MSE, compare probabilities to x in [0,1]
        probs = jax.nn.sigmoid(xhat)
        mse = jnp.mean(jnp.mean((x_batch - probs) ** 2, axis=1))
        # NLL = - E[log p(x|z=μ)] (sum over dims, then mean over batch)
        recon_ll = _bernoulli_loglik_with_logits(x_batch, xhat)   # (B,)
        recon_nll = -jnp.mean(recon_ll)
    elif likelihood == "gaussian":
        mse = jnp.mean(jnp.mean((x_batch - xhat) ** 2, axis=1))
        recon_ll = _gaussian_loglik(x_batch, xhat, sigma_x)       # (B,)
        recon_nll = -jnp.mean(recon_ll)
    else:
        raise ValueError("likelihood must be 'bernoulli_logits' or 'gaussian'.")

    return {"recon_mse": mse, "recon_nll": recon_nll}

def compute_accuracy(
    model,
    x_batch,
    *,
    epsilon: float = 0.01,
    likelihood: str = "bernoulli_logits",
    sigma_x: float = 0.1
):
    """
    Threshold accuracy: fraction of samples with per-sample MSE < ε (deterministic).
    Uses probabilities for Bernoulli and mean for Gaussian.
    """
    xhat = reconstruct_deterministic(model, x_batch)
    if likelihood == "bernoulli_logits":
        xhat = jax.nn.sigmoid(xhat)
    # Gaussian branch already in data space
    mse_per_sample = jnp.mean((x_batch - xhat) ** 2, axis=1)
    return jnp.mean(mse_per_sample < epsilon)

# -----------------------------
# Main evaluation routine
# -----------------------------
def evaluate_model(
    model,
    testloader,
    *,
    loss_fn,                        # pass functools.partial(loss2_VAE, iwae=..., K=..., likelihood=..., ...)
    likelihood: str = "bernoulli_logits",
    sigma_x: float = 0.1,
    epsilon: float = 0.01,
    rng_key: jax.random.PRNGKey = None
):
    """
    Evaluates on a dataloader.
    Computes:
      - avg_objective: mean of your chosen objective over batches
          * ELBO (β-VAE loss) if loss_fn is partial(..., iwae=False, ...)
          * IWAE negative bound (because we minimize) if partial(..., iwae=True, ...)
      - recon_mse_mean: average deterministic reconstruction MSE
      - recon_nll_mean: average deterministic reconstruction NLL
      - acc_thresh: threshold accuracy (MSE < ε)

    Returns: dict with the above metrics.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    params, static = eqx.partition(model, eqx.is_array)

    total_obj = 0.0
    total_mse = 0.0
    total_nll = 0.0
    total_acc = 0.0
    n_batches = 0

    for xb, _ in testloader:
        x = _as_jnp_batch(xb, input_dim=model.input_dim)

        # Objective on this batch (ELBO or IWAE depending on how you partially applied loss_fn)
        rng_key, k = jax.random.split(rng_key)
        batch_obj = loss_fn(params, static, x, k)       # scalar
        total_obj += float(batch_obj)

        # Deterministic reconstruction metrics
        mets = compute_recon_metrics(model, x, likelihood=likelihood, sigma_x=sigma_x)
        total_mse += float(mets["recon_mse"])
        total_nll += float(mets["recon_nll"])

        # Threshold accuracy
        acc = compute_accuracy(model, x, epsilon=epsilon, likelihood=likelihood, sigma_x=sigma_x)
        total_acc += float(acc)

        n_batches += 1

    return {
        "avg_objective": total_obj / n_batches,
        "recon_mse_mean": total_mse / n_batches,
        "recon_nll_mean": total_nll / n_batches,
        "acc_thresh": total_acc / n_batches,
        "batches": n_batches,
    }

# -----------------------------
# Backward-compatible thin wrapper
# (mimics your old evaluate() -> (avg_loss, avg_accuracy))
# -----------------------------
def evaluate(
    model,
    testloader,
    *,
    loss_fn,
    likelihood: str = "bernoulli_logits",
    sigma_x: float = 0.1,
    epsilon: float = 0.01,
    rng_key: jax.random.PRNGKey = None
):
    """
    Backward-compatible wrapper that returns (avg_loss, avg_accuracy).
    Internally uses evaluate_model for richer metrics.
    """
    res = evaluate_model(
        model,
        testloader,
        loss_fn=loss_fn,
        likelihood=likelihood,
        sigma_x=sigma_x,
        epsilon=epsilon,
        rng_key=rng_key,
    )
    return res["avg_objective"], res["acc_thresh"]

            
def train_VAE(
    model: eqx.Module,
    loss_fn,                     # e.g., functools.partial(loss2_VAE, iwae=True/False, ...)
    X,
    Y=None,                      # unused here; kept for API symmetry
    *,
    steps: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    print_every: int = 100,
    key: jax.random.PRNGKey,
    eval_fn=None,                # callable taking (model) or (model, X, Y) — see below
    grad_clip: float
):
    """
    Generic Equinox + Optax trainer that works for both ELBO and IWAE objectives
    (selected via the provided loss_fn).

    - Proper shuffling each epoch (fresh RNG).
    - Per-step PRNG split for stochastic reparameterization.
    - Optional gradient clipping.
    - Eval hook supports either eval_fn(model) or eval_fn(model, X, Y).

    Returns:
        Trained model (recombined from params + static).
    """
    # -----------------
    # Setup
    # -----------------
    key = key if key is not None else jax.random.PRNGKey(0)

    # Ensure JAX arrays (and flatten images if you didn't already)
    X = jnp.array(X, dtype=jnp.float32)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    N = X.shape[0]
    steps_per_epoch = max(1, (N + batch_size - 1) // batch_size)

    # Optimizer (with optional global-norm clipping)
    if grad_clip is not None:
        optim = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adamw(learning_rate),
        )
    else:
        optim = optax.adamw(learning_rate)

    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)

    # -----------------
    # One JIT-compiled step
    # -----------------
    @eqx.filter_jit
    def make_step(params, static, opt_state, x_batch, step_key):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(params, static, x_batch, step_key)
        updates, opt_state_ = optim.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, opt_state_, loss_value

    # -----------------
    # Training loop
    # -----------------
    # initial permutation
    epoch = 0
    ek, key = jax.random.split(key)
    perm = jax.random.permutation(ek, N)

    for step in range(steps):
        # Shuffle at epoch boundaries
        if step % steps_per_epoch == 0 and step > 0:
            epoch += 1
            ek, key = jax.random.split(key)
            perm = jax.random.permutation(ek, N)

        # Slice current minibatch
        start = (step % steps_per_epoch) * batch_size
        end = min(start + batch_size, N)
        idx = perm[start:end]
        x_batch = X[idx]

        # Per-step RNG for stochastic forward pass
        step_key, key = jax.random.split(key)

        # Gradient step
        params, opt_state, loss_val = make_step(params, static, opt_state, x_batch, step_key)

        # Logging / Eval
        if (step % print_every == 0) or (step == steps - 1):
            msg = f"step={step:06d}  loss={float(loss_val):.4f}"

            if eval_fn is not None:
                # Recombine for evaluation (no grad)
                model_eval = eqx.combine(params, static)
                # Support either eval_fn(model) or eval_fn(model, X, Y)
                try:
                    eval_out = eval_fn(model_eval)
                except TypeError:
                    eval_out = eval_fn(model_eval, X, Y)
                # eval_out can be a scalar or dict; format sensibly
                if isinstance(eval_out, dict):
                    parts = [f"{k}={float(v):.4f}" for k, v in eval_out.items() if jnp.ndim(v) == 0]
                    msg += "  |  " + ", ".join(parts)
                else:
                    msg += f"  |  eval={float(eval_out):.4f}"

            print(msg)

    return eqx.combine(params, static)
