# data/csvzarr_dataset.py
from __future__ import annotations
import os, math, json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import pandas as pd

@dataclass
class CsvZarrConfig:
    path: str                           # "D:/data/era5/my_small_1deg.zarr" 或 CSV
    variables_in: List[str]             # 输入变量名
    variables_out: List[str]            # 预测/监督变量名
    history_steps: int = 1              # 用 t-1,… 多少步做输入
    lead_steps: int = 1                 # 预测未来多少步（通常1）
    time_stride: int = 1                # 采样步长
    train_range: Optional[Tuple[str,str]] = None # e.g. ("2016-01-01","2018-06-30")
    valid_range: Optional[Tuple[str,str]] = None # 同上（给 Val 用）
    lat_range: Optional[Tuple[float,float]] = None
    lon_range: Optional[Tuple[float,float]] = None
    normalize: str = "per_var"          # "none" | "per_var"
    stats_path: Optional[str] = None    # 预存均值方差 npz；为空则自动计算并保存到 path 同目录

def _open_dataset(path: str) -> xr.Dataset:
    if path.endswith(".zarr"):
        storage_options = {"token": None} if path.startswith("gs://") else None
        ds = xr.open_zarr(path, storage_options=storage_options)
        return ds
    elif path.endswith(".csv"):
        # 极简 CSV 读取：要求列含 time/valid_time, latitude, longitude + 变量列
        df = pd.read_csv(path)
        tcol = "valid_time" if "valid_time" in df.columns else ("time" if "time" in df.columns else None)
        if tcol is None:
            raise ValueError("CSV must contain 'time' or 'valid_time'.")
        latc = "latitude" if "latitude" in df.columns else "lat"
        lonc = "longitude" if "longitude" in df.columns else "lon"
        df[tcol] = pd.to_datetime(df[tcol], utc=True)
        # 透视为 (time, lat, lon)；仅保留数值列（变量）
        value_cols = [c for c in df.columns if c not in [tcol, latc, lonc, "step", "number", "surface"]]
        idx = df.set_index([tcol, latc, lonc]).sort_index()
        ds = xr.Dataset({c: idx[c].unstack([latc, lonc]).to_xarray() for c in value_cols})
        ds = ds.rename({tcol: "time", latc: "lat", lonc: "lon"}).sortby(["time","lat","lon"])
        return ds
    else:
        raise ValueError(f"Unsupported data path: {path}")

def _subset(ds: xr.Dataset,
            time_range: Optional[Tuple[str,str]],
            lat_range: Optional[Tuple[float,float]],
            lon_range: Optional[Tuple[float,float]]) -> xr.Dataset:
    if time_range:
        ds = ds.sel(time=slice(time_range[0], time_range[1]))
    if lat_range:
        ds = ds.sel(lat=slice(lat_range[0], lat_range[1]))
    if lon_range:
        # 经度可能是 [-180,180] 或 [0,360]；简单兼容处理
        if float(ds.lon.min()) >= 0 and lon_range[0] < 0:
            # 把目标范围映射到 [0,360]
            l0 = (lon_range[0] + 360) % 360
            l1 = (lon_range[1] + 360) % 360
            if l0 <= l1:
                ds = ds.sel(lon=slice(l0, l1))
            else:
                ds = xr.concat([ds.sel(lon=slice(0, l1)), ds.sel(lon=slice(l0, 360))], dim="lon")
        else:
            ds = ds.sel(lon=slice(lon_range[0], lon_range[1]))
    return ds

def _compute_or_load_stats(ds: xr.Dataset, vars_: List[str], stats_path: str) -> Dict[str, Dict[str,float]]:
    if os.path.exists(stats_path):
        arr = np.load(stats_path, allow_pickle=True)["stats"].item()
        return arr
    stats = {}
    for v in vars_:
        if v not in ds:
            raise KeyError(f"Variable '{v}' not in dataset.")
        x = ds[v].astype("float32").values
        m = np.nanmean(x)
        s = np.nanstd(x) + 1e-6
        stats[v] = {"mean": float(m), "std": float(s)}
    np.savez(stats_path, stats=stats)
    return stats

class CsvZarrERA5(Dataset):
    """
    返回：
      X: [C_in * history_steps, H, W]
      Y: [C_out * lead_steps,    H, W]
    """
    def __init__(self, cfg: CsvZarrConfig, split: str):
        assert split in ("train","valid")
        ds = _open_dataset(cfg.path)
        time_range = cfg.train_range if split=="train" else cfg.valid_range
        ds = _subset(ds, time_range, cfg.lat_range, cfg.lon_range)

        self.variables_in  = cfg.variables_in
        self.variables_out = cfg.variables_out
        self.history_steps = cfg.history_steps
        self.lead_steps    = cfg.lead_steps
        self.time_stride   = cfg.time_stride

        # 统计&归一化
        if cfg.normalize == "per_var":
            stats_file = cfg.stats_path or os.path.join(os.path.dirname(cfg.path), "stats.npz")
            self.stats_in  = _compute_or_load_stats(ds, self.variables_in,  stats_file.replace(".npz", "_in.npz"))
            self.stats_out = _compute_or_load_stats(ds, self.variables_out, stats_file.replace(".npz", "_out.npz"))
        else:
            self.stats_in = {v: {"mean":0.0, "std":1.0} for v in self.variables_in}
            self.stats_out= {v: {"mean":0.0, "std":1.0} for v in self.variables_out}

        # 堆成 (time, C, H, W)
        arr_in  = [((ds[v]-self.stats_in[v]["mean"])/self.stats_in[v]["std"]).astype("float32").values for v in self.variables_in]
        arr_out = [((ds[v]-self.stats_out[v]["mean"])/self.stats_out[v]["std"]).astype("float32").values for v in self.variables_out]
        self.X = np.stack(arr_in,  axis=1)   # (T, Cin, H, W)
        self.Y = np.stack(arr_out, axis=1)   # (T, Cout, H, W)

        self.T  = self.X.shape[0]
        self.HW = self.X.shape[2:]
        # 可采样的最后时间索引
        self.max_idx = self.T - (self.history_steps + self.lead_steps)
        self.indices = np.arange(0, self.max_idx+1, self.time_stride, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t0 = self.indices[i]
        x = self.X[t0 : t0 + self.history_steps]      # (H, Cin, H, W)? 现在是 (steps, Cin, H, W)
        y = self.Y[t0 + self.history_steps : t0 + self.history_steps + self.lead_steps]
        # 合并时间到通道： [steps*Cin, H, W]
        x = np.transpose(x, (1,0,2,3)).reshape(-1, *self.HW)
        y = np.transpose(y, (1,0,2,3)).reshape(-1, *self.HW)
        return torch.from_numpy(x), torch.from_numpy(y)
