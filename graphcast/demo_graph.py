from pathlib import Path

rmse_file = Path("outputs/mini_rollout_rmse.txt")
csv_file = Path("outputs/mini_rollout_rmse.csv")

steps = []
vals = []
with rmse_file.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        left, right = line.split(":")
        step = int(left.split()[-1])
        rmse = float(right.strip())
        steps.append(step)
        vals.append(rmse)

with csv_file.open("w") as f:
    f.write("step,rmse\n")
    for s, v in zip(steps, vals):
        f.write(f"{s},{v}\n")

print("wrote:", csv_file)
