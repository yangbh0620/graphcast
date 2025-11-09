from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

PRED_PATH = Path("outputs/graphcast_predictions.nc")
DATASET_PATH = list(Path("assets").glob("*.nc"))[0]

def main():
    preds = xr.load_dataset(PRED_PATH)
    data = xr.load_dataset(DATASET_PATH)

    var = "z"
    level = 500

    pred_z = preds[var].sel(level=level)
    true_z = data["targets"][var].sel(level=level)

    diff = (pred_z - true_z) ** 2
    rmse_time = np.sqrt(diff.mean(dim=("lat", "lon")))

    plt.figure()
    rmse_time.plot(marker="o")
    plt.title("GraphCast RMSE for Z500 (demo data)")
    plt.xlabel("lead time index")
    plt.ylabel("RMSE")
    out = Path("outputs/z500_rmse.png")
    plt.savefig(out, dpi=150)
    print(f"saved plot to {out}")

if __name__ == "__main__":
    main()
