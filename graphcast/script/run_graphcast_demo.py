from pathlib import Path
import jax
import xarray as xr
from huggingface_hub import HfFileSystem, hf_hub_download

from graphcast import checkpoint, graphcast, rollout

ASSETS = Path("assets")
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

def latest_dataset_path() -> str:
    fs = HfFileSystem()
    files = [p for p in fs.ls("datasets/shermansiu/dm_graphcast_datasets", detail=False)
             if p.startswith("datasets/shermansiu/dm_graphcast_datasets/dataset/")]
    first = files[0].rsplit("/", 1)[1]
    return hf_hub_download(
        repo_id="shermansiu/dm_graphcast_datasets",
        repo_type="dataset",
        filename=f"dataset/{first}",
        local_dir=str(ASSETS),
        local_dir_use_symlinks=False,
    )

def load_example_batch():
    ds_path = latest_dataset_path()
    ds = xr.load_dataset(ds_path).compute()
    return ds

def load_graphcast_checkpoint():
    fs = HfFileSystem()
    files = [p for p in fs.ls("models/shermansiu/dm_graphcast", detail=False)
             if p.endswith(".npz")]
    files.sort(key=len, reverse=True)
    fname = files[0].rsplit("/", 1)[1]
    ckpt_path = hf_hub_download(
        repo_id="shermansiu/dm_graphcast",
        filename=fname,
        local_dir=str(ASSETS),
        local_dir_use_symlinks=False,
    )
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return ckpt

def build_predict_fn(ckpt):
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    @jax.jit
    def run_forward(inputs, targets_template, forcings):
        predictor = graphcast.construct_wrapped_graphcast(
            model_config=model_config,
            task_config=task_config,
        )
        preds, _ = predictor.apply(
            params,
            state,
            inputs,
            targets_template=targets_template,
            forcings=forcings,
        )
        return preds

    return run_forward

def main():
    example_batch = load_example_batch()
    ckpt = load_graphcast_checkpoint()
    run_forward = build_predict_fn(ckpt)

    inputs = example_batch["inputs"]
    targets = example_batch["targets"]
    forcings = example_batch["forcings"]

    predictions = rollout.chunked_prediction(
        run_forward,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * 0.0,
        forcings=forcings,
    )

    out_path = OUTDIR / "graphcast_predictions.nc"
    predictions.to_netcdf(out_path)
    print(f"saved predictions to {out_path}")

if __name__ == "__main__":
    main()
