from VAE_IWAE import *
import math
import functools as ft
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Import your objects from the module where you defined them
# Adjust the import path to your file name, e.g.:
# from VAE_IWAE import DeepVAE, loss2_VAE, evaluate_model, train_VAE
from VAE_IWAE import DeepVAE, loss2_VAE, evaluate_model, train_VAE, reconstruct_deterministic

# ---------------------------
# Helpers for tests
# ---------------------------

def make_toy_model_and_data(seed=0, B=64, D=16, L=4, bernoulli=False):
    """Create a small model + synthetic batch."""
    key = jax.random.PRNGKey(seed)
    model = DeepVAE(
        key=key,
        input_dim=D,
        latent_dim=L,
        encoder_hidden=(32, 16),
        decoder_hidden=(16, 32),
    )
    if bernoulli:
        # Data in [0,1]
        X = jax.random.uniform(key, (B, D), minval=0.0, maxval=1.0)
    else:
        # Gaussian-ish data, zero mean, unit-ish variance
        X = jax.random.normal(key, (B, D)) * 0.5
    return model, X


def finite_scalar(x):
    return jnp.isfinite(x).item() and jnp.ndim(x) == 0


# ---------------------------
# 1) Basic smoke tests
# ---------------------------

def test_elbo_gaussian_smoke_and_finite():
    model, X = make_toy_model_and_data(B=32, D=8, L=3, bernoulli=False)
    params, static = eqx.partition(model, eqx.is_array)
    key = jax.random.PRNGKey(1)

    # Configure ELBO with Gaussian likelihood
    loss_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="gaussian",
        sigma_x=1.0,    # any positive value
        beta=0.6,
        alpha=0.0
    )
    loss_val = loss_fn(params, static, X, key)
    assert finite_scalar(loss_val) 
    assert loss_val is not None
    assert loss_val < 100000


def test_elbo_bernoulli_smoke_and_finite():
    model, X = make_toy_model_and_data(B=32, D=8, L=3, bernoulli=True)
    params, static = eqx.partition(model, eqx.is_array)
    key = jax.random.PRNGKey(2)

    # Configure ELBO with Bernoulli-with-logits
    loss_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="bernoulli_logits",
        beta=0.6,
        alpha=0.0
    )
    loss_val = loss_fn(params, static, X, key)
    assert finite_scalar(loss_val)


def test_iwae_smoke_and_finite():
    model, X = make_toy_model_and_data(B=16, D=8, L=3, bernoulli=True)
    params, static = eqx.partition(model, eqx.is_array)
    key = jax.random.PRNGKey(3)

    # IWAE with K>1
    loss_fn = ft.partial(
        loss2_VAE,
        iwae=True,
        K=5,
        likelihood="bernoulli_logits",
        alpha=0.0
    )
    loss_val = loss_fn(params, static, X, key)
    assert finite_scalar(loss_val)


# ---------------------------
# 2) ELBO vs IWAE(K=1): close in expectation
#    (Average over multiple RNG keys to reduce MC noise.)
# ---------------------------

def test_elbo_vs_iwae_K1_close_on_average():
    model, X = make_toy_model_and_data(B=82, D=10, L=4, bernoulli=True)
    params, static = eqx.partition(model, eqx.is_array)

    elbo_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="bernoulli_logits",
        beta=1.0,   # standard ELBO (β=1)
        alpha=0.0
    )
    iwae1_fn = ft.partial(
        loss2_VAE,
        iwae=True,
        K=1,   # single-sample IWAE equals ELBO in expectation
        likelihood="bernoulli_logits",
        alpha=0.0
    )

    # Average over multiple seeds (empirical expectation)
    keys = jax.random.split(jax.random.PRNGKey(10), 200)
    elbo_vals = []
    iwae_vals = []
    for k in keys:
        elbo_vals.append(float(elbo_fn(params, static, X, k)))
        iwae_vals.append(float(iwae1_fn(params, static, X, k)))
    diff = abs(sum(elbo_vals)/len(elbo_vals) - sum(iwae_vals)/len(iwae_vals))
    assert diff < 1e-1  # within small tolerance


# ---------------------------
# 3) IWAE monotonicity in K (on average)
# ---------------------------

def test_iwae_bound_increases_with_K_on_average():
    model, X = make_toy_model_and_data(B=32, D=12, L=4, bernoulli=True)
    params, static = eqx.partition(model, eqx.is_array)

    def avg_negative_bound(K, trials=200):
        # Remember: loss2_VAE returns "- IWAE_bound + alpha*L2" -> we want the bound itself
        fn = ft.partial(loss2_VAE, iwae=True, K=K, likelihood="bernoulli_logits", alpha=0.0)
        keys = jax.random.split(jax.random.PRNGKey(1234 + K), trials)
        vals = []
        for k in keys:
            loss = fn(params, static, X, k)
            vals.append(-float(loss))  # negative of loss = bound
        return sum(vals)/len(vals)

    b1  = avg_negative_bound(1)
    b5  = avg_negative_bound(5)
    b20 = avg_negative_bound(20)

    # Monotone up to MC noise
    assert b5  >= b1  - 1e-3
    assert b20 >= b5  - 1e-3


# ---------------------------
# 4) Deterministic evaluation metrics (recon) with fixed RNG
# ---------------------------

def test_evaluate_model_determinism_with_fixed_rng():
    model, X = make_toy_model_and_data(B=48, D=10, L=3, bernoulli=True)
    # simple dataloader-like iterable: two batches
    loader = [ (X[:24], None), (X[24:], None) ]

    # ELBO eval
    eval_loss_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="bernoulli_logits",
        beta=1.0,
        alpha=0.0
    )

    rng = jax.random.PRNGKey(99)
    res1 = evaluate_model(model, loader, loss_fn=eval_loss_fn,
                          likelihood="bernoulli_logits", rng_key=rng)
    rng2 = jax.random.PRNGKey(99)
    res2 = evaluate_model(model, loader, loss_fn=eval_loss_fn,
                          likelihood="bernoulli_logits", rng_key=rng2)

    # Deterministic with same rng_key
    assert abs(res1["recon_mse_mean"] - res2["recon_mse_mean"]) < 1e-12
    assert abs(res1["recon_nll_mean"] - res2["recon_nll_mean"]) < 1e-12
    # avg_objective uses the same rng as well, so it should match exactly here
    assert abs(res1["avg_objective"] - res2["avg_objective"]) < 1e-12


# ---------------------------
# 5) Tiny training runs actually reduce the objective
# ---------------------------

def test_tiny_training_reduces_elbo_objective():
    model, X = make_toy_model_and_data(B=128, D=16, L=4, bernoulli=True)

    # ELBO loss
    train_loss_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="bernoulli_logits",
        beta=1.0,
        alpha=0.0
    )

    # Measure initial loss on a held-out subset
    params0, static0 = eqx.partition(model, eqx.is_array)
    key0 = jax.random.PRNGKey(2024)
    initial = float(train_loss_fn(params0, static0, X[:28], key0))

    # Do a short training burst on the other half
    trained = train_VAE(
        model=model,
        loss_fn=train_loss_fn,
        X=X[100:],                 # "train split"
        steps=200,
        batch_size=100,
        learning_rate=1e-4,
        print_every=100,
        key=jax.random.PRNGKey(777),
        eval_fn=None,
        grad_clip=1.0
    )

    # Evaluate post-training on the same held-out subset
    params1, static1 = eqx.partition(trained, eqx.is_array)
    key1 = jax.random.PRNGKey(2025)
    after = float(train_loss_fn(params1, static1, X[:28], key1))

    # Expect a reduction
    assert after <= initial + 1e-3

def test_tiny_training_reduces_iwae_bound():
    model, X = make_toy_model_and_data(B=128, D=16, L=4, bernoulli=True)

    # IWAE(K=5)
    K = 5
    train_loss_fn = ft.partial(
        loss2_VAE,
        iwae=True,
        K=K,
        likelihood="bernoulli_logits",
        alpha=0.0
    )

    # Evaluate negative bound (since loss returns -bound)
    def avg_bound(m):
        p, s = eqx.partition(m, eqx.is_array)
        # average over a couple rng keys to reduce noise
        keys = jax.random.split(jax.random.PRNGKey(123), 3)
        vals = []
        for k in keys:
            vals.append(-float(train_loss_fn(p, s, X[:64], k)))
        return sum(vals)/len(vals)

    initial_bound = avg_bound(model)

    trained = train_VAE(
        model=model,
        loss_fn=train_loss_fn,
        X=X,
        steps=200,
        batch_size=64,
        learning_rate=3e-3,
        print_every=200,
        key=jax.random.PRNGKey(4242),
        eval_fn=None,
        grad_clip=1.0
    )

    after_bound = avg_bound(trained)

    # Expect IWAE bound to increase (i.e., loss decreases)
    assert after_bound >= initial_bound - 1e-3


# ---------------------------
# 6) Contract sanity: Bernoulli path produces logits
# ---------------------------

def test_reconstruct_deterministic_logits_then_probs():
    model, X = make_toy_model_and_data(B=8, D=10, L=3, bernoulli=True)
    xhat_logits = reconstruct_deterministic(model, X)  # logits
    # Convert to probabilities and ensure they're in [0,1]
    probs = jax.nn.sigmoid(xhat_logits)
    assert jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)


def make_toy_model_and_gaussian_data(seed=0, B=96, D=16, L=4, data_sigma=0.5):
    """Small model + synthetic Gaussian data ~ N(0, data_sigma^2)."""
    key = jax.random.PRNGKey(seed)
    model = DeepVAE(
        key=key,
        input_dim=D,
        latent_dim=L,
        encoder_hidden=(32, 16),
        decoder_hidden=(16, 32),
    )
    X = jax.random.normal(key, (B, D)) * data_sigma
    return model, X

def finite_scalar(x):
    return jnp.isfinite(x).item() and jnp.ndim(x) == 0

# ---------------------------
# 1) Smoke test (Gaussian ELBO)
# ---------------------------

def test_gaussian_elbo_smoke_and_finite():
    model, X = make_toy_model_and_gaussian_data(B=48, D=12, L=3, data_sigma=0.5)
    params, static = eqx.partition(model, eqx.is_array)
    key = jax.random.PRNGKey(1)

    # Choose sigma_x to roughly match data scale to avoid huge constants
    loss_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="gaussian",
        sigma_x=0.5,   # matches data_sigma above
        beta=1.0,
        alpha=0.0
    )
    loss_val = loss_fn(params, static, X, key)
    assert finite_scalar(loss_val)

# ---------------------------
# 2) Tiny training reduces ELBO (train split)
# ---------------------------

def test_gaussian_tiny_training_reduces_elbo_on_train_split():
    # More data for a clearer optimization signal
    model, X = make_toy_model_and_gaussian_data(B=192, D=16, L=4, data_sigma=0.5)

    # Unified ELBO (Gaussian)
    train_loss_fn = ft.partial(
        loss2_VAE,
        iwae=False,
        likelihood="gaussian",
        sigma_x=0.5,  # match data scale
        beta=1.0,
        alpha=0.0
    )

    # Before/after on the *train* split to test optimization (not generalization)
    train_X = X[:128]

    params0, static0 = eqx.partition(model, eqx.is_array)
    eval_key = jax.random.PRNGKey(2024)   # same key before/after to cancel MC noise
    initial = float(train_loss_fn(params0, static0, train_X, eval_key))

    # Train briefly on the same split
    from VAE_IWAE import train_VAE  # imported here to avoid circulars if any
    trained = train_VAE(
        model=model,
        loss_fn=train_loss_fn,
        X=train_X,
        steps=300,              # short but enough to see a drop
        batch_size=64,
        learning_rate=3e-3,     # a bit higher than 1e-4 to get signal
        print_every=100,
        key=jax.random.PRNGKey(777),
        eval_fn=None,
        grad_clip=1.0
    )

    params1, static1 = eqx.partition(trained, eqx.is_array)
    after = float(train_loss_fn(params1, static1, train_X, eval_key))

    # Expect improvement (negative ELBO should go down)
    assert after <= initial

# ---------------------------
# 3) (Optional) ELBO(1) == IWAE(1) when using the same MC estimator
# ---------------------------

def _mc_elbo_1(params, static, x, key, *, sigma_x):
    """One-sample Monte Carlo ELBO for Gaussian likelihood to match IWAE(K=1)."""
    model = eqx.combine(params, static)
    mu, logvar = jax.vmap(model.encode)(x)           # (B, L)
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, shape=mu.shape)
    z = mu + std * eps

    mean = jax.vmap(model.decode)(z)                 # (B, D)

    # log p(x|z) for Gaussian with fixed sigma_x (sum over dims)
    var = sigma_x ** 2
    log_px = -0.5 * (jnp.sum((x - mean) ** 2 / var + jnp.log(2 * jnp.pi * var), axis=1))

    # log p(z) and log q(z|x) at sampled z (sum over dims)
    log_pz = -0.5 * (jnp.sum(jnp.log(2 * jnp.pi) + z**2, axis=1))
    log_qz = -0.5 * (jnp.sum(jnp.log(2 * jnp.pi) + logvar + (z - mu) ** 2 / jnp.exp(logvar), axis=1))

    # Negative ELBO to match loss scale (mean over batch)
    return -jnp.mean(log_px + log_pz - log_qz)

def test_gaussian_mc_elbo_matches_iwae_k1_on_average():
    model, X = make_toy_model_and_gaussian_data(B=64, D=14, L=5, data_sigma=0.5)
    params, static = eqx.partition(model, eqx.is_array)
    sigma_x = 0.5

    mc1 = ft.partial(_mc_elbo_1, sigma_x=sigma_x)
    iwae1 = ft.partial(loss2_VAE, iwae=True, K=1, likelihood="gaussian", sigma_x=sigma_x, alpha=0.0)

    # Use common random numbers and average many trials
    keys = jax.random.split(jax.random.PRNGKey(10), 200)
    mc_vals = jax.vmap(lambda k: mc1(params, static, X, k))(keys)
    iw_vals = jax.vmap(lambda k: iwae1(params, static, X, k))(keys)

    # Both are the same estimator; averages should be extremely close
    diff = float(jnp.abs(mc_vals.mean() - iw_vals.mean()))
    assert diff < 1e-3