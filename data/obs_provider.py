# data/obs_provider.py
from dataclasses import dataclass
import numpy as np
import xarray as xr

@dataclass
class ObsConfig:
    variables: list[str] = None   # e.g. ["t2m","u10","v10","msl"]
    num_stations: int = 32        # K 个伪站点
    seed: int = 0

def pick_stations(ds: xr.Dataset, cfg: ObsConfig):
    rng = np.random.default_rng(cfg.seed)
    lat = ds.lat.values; lon = ds.lon.values
    lat_idx = rng.integers(0, len(lat), size=cfg.num_stations)
    lon_idx = rng.integers(0, len(lon), size=cfg.num_stations)
    return np.stack([lat_idx, lon_idx], axis=1)  # (K,2)

def extract_obs(ds_t: xr.Dataset, stations_ij: np.ndarray, cfg: ObsConfig):
    """从单个时间步 ds.isel(time=t) 提取观测特征向量与掩码."""
    feats = []
    for v in (cfg.variables or list(ds_t.data_vars)):
        arr = ds_t[v].values  # (lat, lon) 或 (lev,lat,lon) 先做 2D 的
        if arr.ndim == 3:     # 有 pressure level 的话先取地面层/某层
            arr = arr[0]
        vals = arr[stations_ij[:,0], stations_ij[:,1]]  # (K,)
        feats.append(vals.astype(np.float32))
    F = np.stack(feats, axis=1)   # (K, V)
    mask = ~np.isnan(F)           # (K, V)
    F = np.nan_to_num(F, copy=False)
    return F, mask.astype(np.float32)  # (K,V), (K,V)
