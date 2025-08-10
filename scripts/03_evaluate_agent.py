import argparse
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO

from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.rl.mimic_env import MimicEnv


def _run_single_episode(env: MimicEnv, model: PPO) -> pd.DataFrame:
    """Run one deterministic episode and return per-step telemetry with x,y,z."""
    obs, _ = env.reset()
    done = False
    logs = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Poll and collect telemetry
        env.sim_manager.base_vehicle.sensors.poll()
        state = env.telemetry_streamer.get_state()
        pos = env.sim_manager.base_vehicle.state.get('pos', [0, 0, 0])

        state['x'] = pos[0]
        state['y'] = pos[1]
        state['z'] = pos[2]
        logs.append(state)
    return pd.DataFrame(logs)


def _bin_average_by_y(mimic_all: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin mimic telemetry by y using target y values as breakpoints.
    For each interval [y_i, y_{i+1}), average mimic samples within that interval.
    If no samples fall in a bin, NaN is used.
    """
    target_sorted = target_df.sort_values('y').reset_index(drop=True)
    y_edges = target_sorted['y'].to_numpy()
    if len(y_edges) < 2:
        raise ValueError("Target telemetry must contain at least two rows with 'y' to form bins.")

    compare_cols = [c for c in target_sorted.columns if c not in ['time', 'x', 'y', 'z']]

    mimic_sorted = mimic_all.sort_values('y').reset_index(drop=True)

    rows = []
    for i in range(len(y_edges) - 1):
        y_left = y_edges[i]
        y_right = y_edges[i + 1]
        y_mid = 0.5 * (y_left + y_right)

        in_bin = mimic_sorted[(mimic_sorted['y'] >= y_left) & (mimic_sorted['y'] < y_right)]
        row = {
            'y_left': y_left,
            'y_right': y_right,
            'y_mid': y_mid,
            'count': int(len(in_bin)),
        }

        for col in compare_cols:
            row[f'target_{col}'] = target_sorted.loc[i, col] if col in target_sorted.columns else np.nan

        for col in compare_cols:
            if col in mimic_sorted.columns and len(in_bin) > 0:
                row[f'mimic_{col}'] = in_bin[col].mean()
            else:
                row[f'mimic_{col}'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows), compare_cols


def evaluate_batched(
    model_path: str,
    total_episodes: int,
    batch_size: int,
    cooldown_seconds: int,
    output_csv_name: str,
    plot_dir_name: str,
):
    """
    Run evaluation in batches: run `batch_size` episodes, close BeamNG for cooldown,
    and repeat until `total_episodes` are completed. Bin mimic telemetry by target y.
    """
    configs = load_configs()
    sim_cfg = configs['sim']

    target_df = pd.read_csv(sim_cfg['target_data_path'])

    all_samples = []

    model = None
    remaining = total_episodes
    batches = math.ceil(total_episodes / batch_size)

    for b in range(batches):
        episodes_this_batch = min(batch_size, remaining)
        remaining -= episodes_this_batch

        sim_manager = None
        env = None
        try:
            print(f"\n=== Batch {b + 1}/{batches}: launching simulator ===")
            sim_manager = SimulationManager(sim_cfg)
            sim_manager.launch()
            sim_manager.setup_scenario(
                vehicle_model=sim_cfg['base_vehicle_model'],
                vehicle_config=sim_cfg.get('base_vehicle_config', None),
                spawn_target=False
            )

            env = MimicEnv(sim_manager=sim_manager, config=configs)

            if model is None:
                model = PPO.load(model_path, env=env)
            else:
                # Reuse loaded model, attach to the new env
                model.set_env(env)

            print(f"Running {episodes_this_batch} episode(s) in this batch...")
            for ep in range(episodes_this_batch):
                ep_df = _run_single_episode(env, model)
                if 'y' not in ep_df.columns:
                    print("Warning: no 'y' column found in telemetry; skipping episode.")
                    continue
                all_samples.append(ep_df)

        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if sim_manager is not None:
                try:
                    sim_manager.close()
                except Exception:
                    pass

        if b < batches - 1 and cooldown_seconds > 0:
            print(f"Cooling down for {cooldown_seconds} seconds...")
            time.sleep(cooldown_seconds)

    if not all_samples:
        raise RuntimeError("No telemetry collected. Evaluation produced no data.")

    mimic_all_df = pd.concat(all_samples, ignore_index=True)

    binned_df, compare_cols = _bin_average_by_y(mimic_all_df, target_df)

    eval_dir = Path(sim_cfg['data_path']) / "evaluation_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = eval_dir / output_csv_name
    binned_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved binned evaluation CSV to: {output_csv_path}")

    plot_dir = Path(sim_cfg['data_path']) / "plots" / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    x = binned_df['y_mid'] if 'y_mid' in binned_df else binned_df['y_left']

    for col in compare_cols:
        plt.figure(figsize=(14, 6))

        plt.plot(x, binned_df[f'target_{col}'], label=f'Target ({col})', color='green', linestyle='--')

        plt.plot(x, binned_df[f'mimic_{col}'], label=f'Mimic avg ({col})', color='blue', alpha=0.9)

        plt.title(f'Y-binned Comparison: {col}', fontsize=14)
        plt.xlabel('y (bin mid)')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)

        plot_path = plot_dir / f'binned_{col}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print("\nEvaluation complete.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CaRL agent with y-binned averaging across episodes.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model .zip file.")
    parser.add_argument('--episodes', type=int, default=10, help="Total number of episodes to run.")
    parser.add_argument('--batch_size', type=int, default=3, help="Number of episodes per batch before cooldown.")
    parser.add_argument('--cooldown_seconds', type=int, default=10, help="Seconds to sleep between batches after closing BeamNG.")
    parser.add_argument('--output_csv', type=str, default='evaluation_binned.csv', help="Output CSV filename for binned results.")
    parser.add_argument('--plot_dir_name', type=str, default='evaluation_binned', help="Subdirectory under data/plots for saved plots.")
    args = parser.parse_args()

    evaluate_batched(
        model_path=args.model_path,
        total_episodes=args.episodes,
        batch_size=args.batch_size,
        cooldown_seconds=args.cooldown_seconds,
        output_csv_name=args.output_csv,
        plot_dir_name=args.plot_dir_name,
    )


if __name__ == "__main__":
    main()