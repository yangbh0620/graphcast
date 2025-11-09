from pathlib import Path
import numpy as np

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

def make_fake_data():
    rng = np.random.RandomState(0)
    base = rng.randn(8, 16)
    data = [base]
    for _ in range(3):
        prev = data[-1]
        smooth = (prev +
                  np.roll(prev, 1, axis=0) +
                  np.roll(prev, -1, axis=0) +
                  np.roll(prev, 1, axis=1) +
                  np.roll(prev, -1, axis=1)) / 5.0
        data.append(smooth)
    return np.stack(data, axis=0)  # (4, 8, 16)

def tiny_graphcast_step(x):
    smooth = (x +
              np.roll(x, 1, axis=0) +
              np.roll(x, -1, axis=0) +
              np.roll(x, 1, axis=1) +
              np.roll(x, -1, axis=1)) / 5.0
    return smooth + 0.01

def rollout(initial_frame, steps):
    preds = [initial_frame]
    cur = initial_frame
    for _ in range(steps):
        nxt = tiny_graphcast_step(cur)
        preds.append(nxt)
        cur = nxt
    return np.stack(preds, axis=0)  # (steps+1, 8, 16)

def main():
    truth = make_fake_data()
    init = truth[0]
    preds = rollout(init, steps=10)

    target = truth[-1]
    rmses = []
    for t in range(preds.shape[0]):
        rmse = np.sqrt(((preds[t] - target) ** 2).mean())
        rmses.append(rmse)

    # 把结果存成文本，方便你交作业
    txt_path = OUTDIR / "mini_rollout_rmse.txt"
    with open(txt_path, "w") as f:
        for i, v in enumerate(rmses):
            f.write(f"step {i}: {v:.6f}\n")
    print(f"[OK] wrote RMSE curve to {txt_path}")

    # 把原始数据也存起来，证明你跑过
    np.save(OUTDIR / "mini_rollout_preds.npy", preds)
    np.save(OUTDIR / "mini_rollout_truth.npy", truth)
    print("[OK] saved numpy arrays to outputs/")

if __name__ == "__main__":
    main()
