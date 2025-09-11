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
#                  VAE Loss Helper Functions
# ============================================================

def vae_loss(
    model: DeepVAE,
    x: Float[Array, "batch input_dim"],
    key: jax.random.PRNGKey
) -> Float[Array, ""]:
    """
    Compute the total VAE loss on a batch:
      total = recon_MSE + beta * KL(q(z|x) || p(z)) + alpha * L2(weights)

    Uses the model's stochastic forward pass via vmap over the batch.
    """
    # Important: split key per-sample so each item samples its own eps
    keys = jax.random.split(key, x.shape[0])
    x_hat, mu, logvar = jax.vmap(lambda xi, ki: model(xi, ki))(x, keys)
    return vae_loss_components(x, x_hat, mu, logvar, model)


def vae_loss_components(
    x: Float[Array, "batch input_dim"],
    x_hat: Float[Array, "batch input_dim"],
    mu: Float[Array, "batch latent_dim"],
    logvar: Float[Array, "batch latent_dim"],
    params,
    *,
    beta: float = 0.5,
    alpha: float = 0,
) -> Float[Array, ""]:
    """
    Break out the loss terms for clarity and tuning.

    Args:
        x: Ground-truth inputs (batch, D).
        x_hat: Reconstructions (batch, D).
        mu, logvar: Encoder outputs (batch, latent_dim).
        params: Model (or a params pytree) for optional L2 regularization.
        beta: Weight for KL term (β-VAE style).
        alpha: Weight for L2 penalty on weights (biases excluded).

    Returns:
        Scalar total loss.
    """
    # 1) Reconstruction (MSE)
    recon_loss = jnp.mean((x - x_hat) ** 2)

    # 2) KL divergence for diagonal Gaussians:
    #    KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) averaged over batch
    kl_loss = -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))

    # 3) Optional L2 on weights only (skip vectors, e.g., biases)
    def is_weight(p):
        return isinstance(p, jnp.ndarray) and p.ndim > 1

    l2_loss = sum(
        jnp.sum(jnp.square(p))
        for p in jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_array))
        if is_weight(p)
    )

    return recon_loss + beta * kl_loss + alpha * l2_loss


def vae_loss(
    model: DeepVAE,
    x: Float[Array, "batch input_dim"],
    key: jax.random.PRNGKey
) -> Float[Array, ""]:
    """
    Computes the total VAE loss (reconstruction + KL + optional L2).

    Args:
        model: Instance of DeepVAE.
        x: Batched input data of shape (batch, input_dim).
        key: PRNGKey for stochastic reparameterization.

    Returns:
        Scalar loss averaged over batch.
    """
    x_hat, mu, logvar = jax.vmap(lambda xi: model(xi, key))(x)
    return vae_loss_components(x, x_hat, mu, logvar, model)


def vae_loss_components(
    x: Float[Array, "batch input_dim"],
    x_hat: Float[Array, "batch input_dim"],
    mu: Float[Array, "batch latent_dim"],
    logvar: Float[Array, "batch latent_dim"],
    params,
    beta: float = 0.6,
    alpha: float = 1e-5
) -> Float[Array, ""]:
    """
    Computes the components of the VAE loss.

    Args:
        x: Ground truth input.
        x_hat: Reconstruction of input.
        mu: Latent mean vectors.
        logvar: Latent log-variance vectors.
        params: Model (or model parameters).
        beta: Weight for KL-divergence.
        alpha: Weight for L2 regularization.

    Returns:
        Scalar total loss.
    """
    # MSE reconstruction loss
    recon_loss = jnp.mean((x - x_hat) ** 2)

    # KL divergence loss (closed form for diagonal Gaussians)
    kl_loss = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

    # L2 regularization on weights (filter out biases and 1D)
    def is_weight(p):
        return isinstance(p, jnp.ndarray) and p.ndim > 1

    l2_loss = sum([
        jnp.sum(jnp.square(p))
        for p in jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_array))
        if is_weight(p)
    ])

    return recon_loss + beta * kl_loss + alpha * l2_loss


def loss2_VAE(params, static, x, key):
    """
    Wrapper for training: recombines model from params and static fields.

    Args:
        params: Trainable PyTree parameters.
        static: Non-trainable PyTree parts of the model.
        x: Batched input data.
        key: PRNG key.

    Returns:
        Scalar loss.
    """
    model = eqx.combine(params, static)
    return vae_loss(model, x, key)

@eqx.filter_jit
def compute_accuracy(
    model: DeepVAE,
    x: Float[Array, "batch input_dim"],
    epsilon: float = 0.01
) -> Float[Array, ""]:
    """
    Computes the proportion of samples with reconstruction MSE < ε.

    Args:
        model: Trained autoencoder model.
        x: Batched input samples.
        epsilon: Threshold to count as "correctly reconstructed".

    Returns:
        Float scalar accuracy in [0, 1].
    """
    pred_x = jax.vmap(model)(x)
    mse_per_sample = jnp.mean((x - pred_x) ** 2, axis=1)
    return jnp.mean(mse_per_sample < epsilon)

def evaluate(
    model: DeepVAE,
    testloader,
    epsilon: float = 0.01
) -> tuple[float, float]:
    """
    Evaluates the model over a test set for loss and accuracy.

    Args:
        model: Trained VAE model.
        testloader: Iterator yielding batches (x, _).
        epsilon: MSE threshold for accuracy.

    Returns:
        Tuple: (average_loss, average_accuracy)
    """
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x_batch, _ in testloader:
        x_batch = jnp.array(x_batch)
        total_loss += vae_loss(model, x_batch, jax.random.PRNGKey(0))  # Dummy key
        total_acc += compute_accuracy(model, x_batch, epsilon)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches

def get_batches(
    X: Float[Array, "n input_dim"],
    Y: Float[Array, "n output_dim"] = None,
    batch_size: int = 128,
    *,
    rng_key=None
):
    """
    Yields batches from X (and optionally Y) in random order.

    Args:
        X: Input dataset.
        Y: Optional target values.
        batch_size: Number of samples per batch.
        rng_key: Optional PRNGKey.

    Yields:
        Tuple of minibatches (xb, yb)
    """
    N = X.shape[0]
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
    while True:
        perm = jax.random.permutation(rng_key, N)
        X = X[perm]
        Y = Y[perm] if Y is not None else None
        for i in range(0, N, batch_size):
            xb = X[i:i + batch_size]
            yb = Y[i:i + batch_size] if Y is not None else None
            yield xb, yb

            
def train_VAE(
    model: eqx.Module,
    loss_fn,
    X,
    Y=None,
    steps: int = 1000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    print_every: int = 100,
    key: jax.random.PRNGKey = None,
    eval_fn=None,
):
    """
    Generic training loop for models in Equinox using Optax.

    Args:
        model: Equinox model.
        loss_fn: Callable loss function.
        X: Input training data.
        Y: Optional labels.
        steps: Number of training steps.
        batch_size: Minibatch size.
        learning_rate: Learning rate for optimizer.
        print_every: Logging frequency.
        key: PRNG key.
        eval_fn: Optional function for validation.

    Returns:
        Trained model (recombined from params + static).
    """
    key = key if key is not None else jax.random.PRNGKey(0)
    optim = optax.adamw(learning_rate)

    # Split parameters into trainable (params) and static (buffers)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)
    batch_gen = get_batches(X, Y, batch_size, rng_key=key)

    @eqx.filter_jit
    def make_step(params, static, opt_state, x, y, step_key):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(params, static, x, step_key)
        updates, opt_state_ = optim.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, opt_state_, loss_value

    for step in range(steps):
        xb, yb = next(batch_gen)
        step_key, key = jax.random.split(key)
        params, opt_state, loss_val = make_step(params, static, opt_state, xb, yb, step_key)

        if step % print_every == 0 or step == steps - 1:
            msg = f"step={step}, loss={loss_val.item():.4f}"
            if eval_fn is not None:
                model_eval = eqx.combine(params, static)
                eval_result = eval_fn(model_eval, X, Y)
                msg += f", eval={eval_result:.4f}"
            print(msg)

    return eqx.combine(params, static)