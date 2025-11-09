from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

def download_dataset_files():
    fs = HfFileSystem()
    files = [p for p in fs.ls("datasets/shermansiu/dm_graphcast_datasets", detail=False)
             if p.startswith("datasets/shermansiu/dm_graphcast_datasets/dataset/")]
    local_paths = []
    for remote in files:
        fname = remote.rsplit("/", 1)[1]
        local = hf_hub_download(
            repo_id="shermansiu/dm_graphcast_datasets",
            repo_type="dataset",
            filename=f"dataset/{fname}",
            local_dir=str(ASSETS),
            local_dir_use_symlinks=False,
        )
        local_paths.append(local)
    return local_paths

def download_model_weights():
    fs = HfFileSystem()
    files = fs.ls("models/shermansiu/dm_graphcast", detail=False)
    npz_files = [f for f in files if f.endswith(".npz")]
    local_weights = []
    for remote in npz_files:
        fname = remote.rsplit("/", 1)[1]
        local = hf_hub_download(
            repo_id="shermansiu/dm_graphcast",
            filename=fname,
            local_dir=str(ASSETS),
            local_dir_use_symlinks=False,
        )
        local_weights.append(local)
    return local_weights

def main():
    print("Downloading example ERA5/HRES datasets from HF ...")
    ds_files = download_dataset_files()
    print(f"Downloaded {len(ds_files)} dataset files into assets/")

    print("Downloading GraphCast weights from HF ...")
    weight_files = download_model_weights()
    print(f"Downloaded {len(weight_files)} weight files into assets/")

if __name__ == "__main__":
    main()
