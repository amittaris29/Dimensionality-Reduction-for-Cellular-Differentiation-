import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import functools as ft
import jax.nn as jnn
from typing import Any
from typing import List


class DeepAutoencoder(eqx.Module):
    
    """
    Deep fully connected autoencoder using Equinox.
    
    This model maps high-dimensional input data to a low-dimensional latent space
    using an encoder, and reconstructs the input from the latent code via a decoder.

    Attributes:
        encoder_layers (List[eqx.nn.Linear]): Linear layers of the encoder.
        decoder_layers (List[eqx.nn.Linear]): Linear layers of the decoder.
        latent_dim (int): Dimension of the encoded latent space.
        input_dim (int): Dimension of the input data.
    """
    
    encoder_layers: List[eqx.nn.Linear]
    decoder_layers: List[eqx.nn.Linear]
    latent_dim: int
    input_dim: int

    def __init__(self, key: jax.random.PRNGKey, latent_dim: int = 2, input_dim: int = 3):
        """
        Initializes the autoencoder model architecture.

        Args:
            key (jax.random.PRNGKey): Random key for layer initialization.
            latent_dim (int): Dimensionality of the latent space.
            input_dim (int): Dimensionality of the input data.
        """
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        k1, k2 = jax.random.split(key)

        # Define encoder and decoder architectures
        enc_sizes = [input_dim, 512, 256, 128, latent_dim]
        dec_sizes = [latent_dim, 128, 256, 512, input_dim]

        # Initialize encoder layers
        self.encoder_layers = [
            eqx.nn.Linear(enc_sizes[i], enc_sizes[i + 1], key=jax.random.fold_in(k1, i))
            for i in range(len(enc_sizes) - 1)
        ]

        # Initialize decoder layers
        self.decoder_layers = [
            eqx.nn.Linear(dec_sizes[i], dec_sizes[i + 1], key=jax.random.fold_in(k2, i))
            for i in range(len(dec_sizes) - 1)
        ]

    def encode(self, x: Float[Array, "input_dim"]) -> Float[Array, "latent_dim"]:
        """
        Encodes the input into a latent representation.

        Args:
            x: Input vector of shape (input_dim,)

        Returns:
            Latent vector of shape (latent_dim,)
        """
        for layer in self.encoder_layers[:-1]:
            x = jnn.relu(layer(x))
        return self.encoder_layers[-1](x)

    def decode(self, z: Float[Array, "latent_dim"]) -> Float[Array, "input_dim"]:
        """
        Decodes a latent vector back to the input space.

        Args:
            z: Latent vector of shape (latent_dim,)

        Returns:
            Reconstructed input of shape (input_dim,)
        """
        for layer in self.decoder_layers[:-1]:
            z = jnn.relu(layer(z))
        return self.decoder_layers[-1](z)

    def reconstruct(self, x: Float[Array, "input_dim"]) -> Float[Array, "input_dim"]:
        """
        Reconstructs input by encoding and decoding it.

        Args:
            x: Input data of shape (input_dim,)

        Returns:
            Reconstructed input of shape (input_dim,)
        """
        return self.decode(self.encode(x))

    def __call__(self, x: Float[Array, "input_dim"]) -> Float[Array, "input_dim"]:
        """
        Calls the model as a function (shorthand for reconstruction).

        Args:
            x: Input data.

        Returns:
            Reconstructed output.
        """
        return self.reconstruct(x)


def loss_AE(
    model: "DeepAutoencoder",
    x: Float[Array, "batch input_dim"]
) -> Float[Array, ""]:
    """
    Computes the total loss for an autoencoder, including reconstruction loss
    and optional L2 weight regularization.

    Args:
        model (DeepAutoencoder): The autoencoder model.
        x (Float[Array, "batch input_dim"]): Input batch of shape (batch, input_dim).

    Returns:
        Float[Array, ""]: The scalar total loss value.
    """
    # Reconstruct the input using the model
    pred_x = jax.vmap(model)(x)
    
    # Compute loss
    return least_squares_autoencoder(x, pred_x, model)


def least_squares_autoencoder(
    x: Float[Array, "batch input_dim"],
    pred_x: Float[Array, "batch input_dim"],
    params
) -> Float[Array, ""]:
    """
    Mean squared reconstruction error + L2 regularization on weights.

    Args:
        x (Float[Array, "batch input_dim"]): Original input batch.
        pred_x (Float[Array, "batch input_dim"]): Reconstructed inputs from the model.
        params: The model object (used for filtering trainable parameters).

    Returns:
        Float[Array, ""]: The total loss (reconstruction + L2).
    """
    # === Reconstruction Loss ===
    recon_loss = jnp.mean((x - pred_x) ** 2)

    # === L2 Regularization ===
    def is_weight(p):
        """Filter to apply L2 only on weight matrices, not biases."""
        return isinstance(p, jnp.ndarray) and p.ndim > 1

    l2_loss = sum(
        jnp.sum(jnp.square(p))
        for p in jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_array))
        if is_weight(p)
    )

    total_loss = recon_loss + 1e-5 * l2_loss
    return total_loss


def loss2_AE(
    params,
    static,
    x: Float[Array, "batch input_dim"]
) -> Float[Array, ""]:
    """
    Loss wrapper for use with Equinox optimizers, combining params and static
    parts of the model.

    Args:
        params: Trainable parameters of the model.
        static: Non-trainable/static parts of the model.
        x (Float[Array, "batch input_dim"]): Input batch.

    Returns:
        Float[Array, ""]: The scalar total loss.
    """
    model = eqx.combine(params, static)
    return loss_AE(model, x)


@eqx.filter_jit
def compute_accuracy(
    model,
    x: Float[Array, "batch input_dim"]
) -> Float[Array, ""]:
    """
    Computes the fraction of inputs with MSE reconstruction error below a threshold ε.

    Args:
        model: Trained autoencoder model.
        x (Float[Array, "batch input_dim"]): Batch of input data.

    Returns:
        Float[Array, ""]: Scalar accuracy (between 0 and 1).
    """
    pred_x = jax.vmap(model)(x)
    mse_per_sample = jnp.mean((x - pred_x) ** 2, axis=1)
    epsilon = 0.01  # Threshold for "accurate" reconstruction
    return jnp.mean(mse_per_sample < epsilon)

def evaluate(model, testloader):
    """
    Evaluates the model on a testloader by computing average loss and accuracy.

    Args:
        model: Trained autoencoder model.
        testloader: Iterable yielding batches (x, _) where labels are unused.

    Returns:
        Tuple[Float, Float]: Mean loss and accuracy across batches.
    """
    total_loss = 0
    total_acc = 0
    n_batches = 0

    for x, _ in testloader:  # labels not needed
        x = jnp.array(x)
        total_loss += loss_AE(model, x)
        total_acc += compute_accuracy(model, x)
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
          
        
def train_AE(
    model,
    loss_fn,
    X,
    Y=None,
    steps=1000,
    batch_size=128,
    learning_rate=1e-3,
    print_every=100,
    key=None,
    eval_fn=None,
):
    """
    Trains an Equinox model using a given loss function and optimizer.

    Args:
        model: The model to be trained.
        loss_fn: A function of the form loss(model, x) or loss(params, static, x).
        X: Training input data.
        Y: Optional labels (ignored in typical autoencoder).
        steps: Total training steps.
        batch_size: Mini-batch size.
        learning_rate: Learning rate for AdamW.
        print_every: Frequency of logging.
        key: PRNG key.
        eval_fn: Optional evaluation function like `evaluate`.

    Returns:
        Trained model (with updated parameters).
    """
    key = key or jax.random.PRNGKey(0)
    optim = optax.adamw(learning_rate)

    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)
    batch_gen = get_batches(X, Y, batch_size, rng_key=key)

    @eqx.filter_jit
    def make_step(params, static, opt_state, x, y):
        model = eqx.combine(params, static)
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(params, static, x)
        updates, opt_state_ = optim.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, opt_state_, loss_value

    for step in range(steps):
        xb, yb = next(batch_gen)
        params, opt_state, loss_val = make_step(params, static, opt_state, xb, yb)

        if step % print_every == 0 or step == steps - 1:
            msg = f"step={step}, loss={loss_val.item():.4f}"
            if eval_fn is not None:
                model_eval = eqx.combine(params, static)
                eval_result = eval_fn(model_eval, X, Y)
                msg += f", eval={eval_result:.4f}"
            print(msg)

    return eqx.combine(params, static)