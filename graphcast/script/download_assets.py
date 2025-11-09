# scripts/download_assets.py
from pathlib import Path
from huggingface_hub import hf_hub_download

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

def download_one_dataset():
    # 你可以去 https://huggingface.co/datasets/shermansiu/dm_graphcast_datasets 看具体名字
    # 我这里先举一个常见名字，下载不到就把下面这一行的文件名改成网页上的
    filename = "dataset/example_20170101_0000.nc"
    local = hf_hub_download(
        repo_id="shermansiu/dm_graphcast_datasets",
        repo_type="dataset",
        filename=filename,
        local_dir=str(ASSETS),
        local_dir_use_symlinks=False,
    )
    return [local]

def download_model_weights():
    weight_file = (
        "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    )
    local = hf_hub_download(
        repo_id="shermansiu/dm_graphcast",
        filename=weight_file,
        local_dir=str(ASSETS),
        local_dir_use_symlinks=False,
    )
    return [local]

def main():
    print("Downloading example dataset ...")
    ds = download_one_dataset()
    print("Downloaded dataset files:", ds)

    print("Downloading GraphCast weights ...")
    ws = download_model_weights()
    print("Downloaded weight files:", ws)

if __name__ == "__main__":
    main()
