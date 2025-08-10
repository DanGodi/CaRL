from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # Try to match columns case-insensitively
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot (training reward) × (fps).")
    parser.add_argument("--reward_csv", type=str, default="training_reward.csv", help="Path to the reward CSV.")
    parser.add_argument("--fps_csv", type=str, default="training_fps.csv", help="Path to the FPS CSV.")
    parser.add_argument("--out_png", type=str, default=str(Path("data/plots/reward_times_fps.png")), help="Output plot path.")
    parser.add_argument("--out_csv", type=str, default=str(Path("data/plots/reward_times_fps.csv")), help="Output CSV path.")
    args = parser.parse_args()

    reward_path = Path(args.reward_csv)
    fps_path = Path(args.fps_csv)

    rdf = pd.read_csv(reward_path)
    fdf = pd.read_csv(fps_path)

    # Try to detect step and value columns
    step_candidates = ["timesteps", "timestep", "steps", "step", "global_step", "Step"]
    reward_candidates = ["reward", "value", "mean_reward", "episode_reward", "ep_rew_mean", "train_reward", "rollout/ep_rew_mean"]
    fps_candidates = ["Value"]

    r_step = pick_col(rdf, step_candidates)
    r_val  = pick_col(rdf, reward_candidates) or pick_col(rdf, ["value"])
    f_step = pick_col(fdf, step_candidates)
    f_val  = pick_col(fdf, fps_candidates)

    if r_val is None:
        raise RuntimeError(f"Could not find a reward column in {reward_path}. Columns: {list(rdf.columns)}")
    if f_val is None:
        raise RuntimeError(f"Could not find an fps column in {fps_path}. Columns: {list(fdf.columns)}")

    # Align data
    if r_step is not None and f_step is not None:
        rdf_sorted = rdf[[r_step, r_val]].dropna().sort_values(r_step).rename(columns={r_step: "step", r_val: "reward"})
        fdf_sorted = fdf[[f_step, f_val]].dropna().sort_values(f_step).rename(columns={f_step: "step", f_val: "fps"})
        merged = pd.merge_asof(rdf_sorted, fdf_sorted, on="step", direction="nearest")
    else:
        # Fallback: align by index
        n = min(len(rdf), len(fdf))
        rdf_trim = rdf.iloc[:n].reset_index(drop=True)
        fdf_trim = fdf.iloc[:n].reset_index(drop=True)
        merged = pd.DataFrame({
            "step": rdf_trim.index,
            "reward": rdf_trim[r_val],
            "fps": fdf_trim[f_val],
        })

    merged["reward_x_fps"] = merged["reward"] / merged["fps"]

    # Ensure output dir exists
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    merged.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(merged["step"], merged["reward_x_fps"], label="reward × fps", color="royalblue")
    plt.title("FPS independent Reward")
    plt.xlabel("step")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    out_pdf = out_png.with_suffix('.pdf')
    plt.savefig(out_pdf)
    plt.close()

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")

if __name__ == "__main__":
    main()