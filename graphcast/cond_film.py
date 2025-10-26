# graphcast/cond_film.py
import haiku as hk
import jax.numpy as jnp

class FiLM(hk.Module):
    """Feature-wise Linear Modulation from a conditioning vector."""
    def __init__(self, channels: int, name=None):
        super().__init__(name or "FiLM")
        self.channels = channels

    def __call__(self, h: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        # h: (..., C), cond: (d,)
        gamma = hk.Linear(self.channels, name="gamma")(cond)  # (C,)
        beta  = hk.Linear(self.channels, name="beta")(cond)   # (C,)
        return h * (1.0 + gamma) + beta
