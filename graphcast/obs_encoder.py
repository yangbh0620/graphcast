# graphcast/obs_encoder.py
from typing import Optional
import haiku as hk
import jax.numpy as jnp

class ObsEncoder(hk.Module):
    """Encodes sparse station observations (K,V) (+ mask) -> global latent."""
    def __init__(self, d_model: int = 128, name: Optional[str] = None):
        super().__init__(name=name or "ObsEncoder")
        self.d_model = d_model

    def __call__(self, obs_feats: jnp.ndarray, obs_mask: jnp.ndarray) -> jnp.ndarray:
        # obs_feats: (K,V), obs_mask: (K,V)
        x = jnp.concatenate([obs_feats, obs_mask], axis=-1)        # (K,2V)
        mlp = hk.nets.MLP([256, 256, self.d_model], activate_final=False)
        x = mlp(x)                                                 # (K, d)
        # masked mean pooling
        w = jnp.clip(jnp.sum(obs_mask, axis=-1, keepdims=True), 1.0, None)  # (K,1)
        x = jnp.sum(x, axis=0, keepdims=False) / jnp.sum((w>0).astype(x.dtype))  # (d,)
        return x
