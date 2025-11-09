from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

def main():
    fs = HfFileSystem()

    # 1. 列出数据集里真正存在的文件
    files = fs.ls(
        "datasets/shermansiu/dm_graphcast_datasets/dataset",
        detail=False
    )
    first_remote = files[0]
    real_name = first_remote.rsplit("/", 1)[1]
    print("Will download dataset file:", real_name)

    # 2. 下载这个 .nc
    local_ds = hf_hub_download(
        repo_id="shermansiu/dm_graphcast_datasets",
        repo_type="dataset",
        filename=f"dataset/{real_name}",
        local_dir=str(ASSETS),
        local_dir_use_symlinks=False,
    )
    print("Dataset saved to:", local_ds)

    # 3. 下载模型权重
    model_files = fs.ls("models/shermansiu/dm_graphcast", detail=False)
    npz_files = [f for f in model_files if f.endswith(".npz")]
    npz_files.sort(key=len, reverse=True)
    weight_remote = npz_files[0]
    weight_name = weight_remote.rsplit("/", 1)[1]
    print("Will download model weight:", weight_name)

    local_wt = hf_hub_download(
        repo_id="shermansiu/dm_graphcast",
        filename=weight_name,
        local_dir=str(ASSETS),
        local_dir_use_symlinks=False,
    )
    print("Weight saved to:", local_wt)
    print("All done.")

if __name__ == "__main__":
    main()
