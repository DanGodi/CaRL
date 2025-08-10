import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager  # noqa: F401
from carlpack.rl.mimic_env import MimicEnv  # noqa: F401


def _load_target_path(sim_cfg):
    target_csv = Path(sim_cfg['target_data_path'])
    if not target_csv.is_file():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")
    df = pd.read_csv(target_csv)
    # Build AI script (x,y,z,t). Ensure required columns exist.
    for c in ['x', 'y', 'z', 'time']:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in target CSV: {target_csv}")
    path_script = [
        {'x': r['x'], 'y': r['y'], 'z': r['z'], 't': r['time']} for _, r in df.iterrows()
    ]
    max_time = float(df['time'].iloc[-1])
    last_pos = np.array([df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1]])
    return path_script, max_time, last_pos


def _run_ai_script_until_end(sim_manager, vehicle, script, last_pos, time_limit=None, completion_threshold=5.0):
    # Teleport to start and set AI script
    start_pos = (script[0]['x'], script[0]['y'], script[0]['z'])
    vehicle.teleport(start_pos, rot_quat=(0, 0, 1, 0), reset=True)
    vehicle.ai.set_mode('script')
    vehicle.ai.set_script(script)

    sim_manager.bng.resume()
    steps = 0
    max_steps = int((time_limit or 60.0) * 120) if time_limit else 120 * 120  # hard guard
    try:
        while True:
            vehicle.sensors.poll()
            pos = vehicle.state.get('pos', [0, 0, 0])
            pos_vec = np.array([pos[0], pos[1], pos[2]])
            dist_to_end = np.linalg.norm(pos_vec - last_pos)

            if dist_to_end < completion_threshold:
                break

            steps += 1
            if time_limit is not None and steps >= int(time_limit * 60):
                # soft stop if we exceeded target time by a margin
                break
            if steps >= max_steps:
                # hard failsafe
                break

            time.sleep(1 / 50.0)
    finally:
        sim_manager.bng.pause()


def _find_latest_model_path(sim_cfg):
    checkpoints_dir = Path(sim_cfg['model_save_path']) / "checkpoints"
    final_model = Path(sim_cfg['model_save_path']) / "ppo_carl_final.zip"

    candidates = []
    if checkpoints_dir.is_dir():
        for p in checkpoints_dir.glob("*.zip"):
            candidates.append(p)
    if final_model.is_file():
        candidates.append(final_model)

    if not candidates:
        raise FileNotFoundError(f"No model files found in {checkpoints_dir} or {final_model}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    configs = load_configs()
    sim_cfg = configs['sim']

    path_script, max_time, last_pos = _load_target_path(sim_cfg)

    sim_manager = None
    try:
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch()

        # 1) Target truck runs the course
        print("\n=== Running target truck on recorded course ===")
        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['target_vehicle_model'],
            vehicle_config=sim_cfg.get('target_vehicle_config', None),
            spawn_target=False
        )
        _run_ai_script_until_end(
            sim_manager,
            sim_manager.base_vehicle,
            path_script,
            last_pos,
            time_limit=max_time + 5.0,  # allow a small buffer
            completion_threshold=5.0
        )
        print("Target truck run complete.")

        # 2) Mimic car with random step actions runs the course
        print("\n=== Running mimic car with RANDOM-STEP actions ===")
        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['base_vehicle_model'],
            vehicle_config=sim_cfg.get('base_vehicle_config', None),
            spawn_target=False
        )

        start_pos = (path_script[0]['x'], path_script[0]['y'], path_script[0]['z'])
        v = sim_manager.base_vehicle
        v.teleport(start_pos, rot_quat=(0, 0, 1, 0), reset=True)
        v.ai.set_mode('script')
        v.ai.set_script(path_script)
        sim_manager.bng.resume()

        # Sample control parameters uniformly within configured ranges each step
        action_map = configs['env']['action_parameter_mappings']  # from env_config.yaml

        steps = 0
        try:
            while True:
                v.sensors.poll()
                pos = v.state.get('pos', [0, 0, 0])
                pos_vec = np.array([pos[0], pos[1], pos[2]])
                dist_to_end = np.linalg.norm(pos_vec - last_pos)

                # Random-step baseline: new random control each tick
                random_params = {
                    key: float(np.random.uniform(p['min'], p['max']))
                    for key, p in action_map.items()
                }
                sim_manager.apply_vehicle_controls(random_params)

                if dist_to_end < 5.0:
                    break

                steps += 1
                if steps >= int((max_time + 5.0) * 60):
                    break

                time.sleep(1 / 50.0)
        finally:
            sim_manager.bng.pause()
        print("Mimic car (random-step) run complete.")

        # 3) Mimic car with the most updated PPO model
        print("\n=== Running mimic car with latest PPO model ===")

        # Important: do NOT pre-setup the scenario here; MimicEnv.reset() will handle it.
        env = MimicEnv(sim_manager=sim_manager, config=configs)
        latest_model_path = _find_latest_model_path(sim_cfg)
        print(f"Loading model: {latest_model_path}")
        model = PPO.load(latest_model_path, device="auto")

        obs, _ = env.reset()
        done = False

        # Safety cap to prevent infinite loops
        max_steps = int((max_time + 5.0) * 120)
        steps = 0

        sim_manager.bng.resume()
        try:
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                time.sleep(1 / 50.0)
        finally:
            sim_manager.bng.pause()

        print("Mimic car (PPO) run complete.")

    finally:
        if sim_manager:
            sim_manager.close()


if __name__ == "__main__":
    main()