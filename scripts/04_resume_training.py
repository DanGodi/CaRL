import os
import re
import glob
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.rl.mimic_env import MimicEnv


def find_latest_checkpoint(checkpoints_dir: str, prefix: str = "ppo_carl") -> str | None:
    """
    Find the latest checkpoint (highest step count) in the given directory.
    """
    pattern = os.path.join(checkpoints_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None

    def extract_steps(path: str) -> int:
        m = re.search(rf"{re.escape(prefix)}_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    files.sort(key=extract_steps)
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="Resume PPO training from the latest or a specified checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific checkpoint .zip to resume from.")
    parser.add_argument("--timesteps", type=int, default=None, help="Additional timesteps to train.")
    parser.add_argument("--save-name", type=str, default="ppo_carl_resumed", help="Base filename for the resumed model.")
    parser.add_argument("--tb-name", type=str, default="PPO_CaRL_Resume", help="TensorBoard run name.")
    args = parser.parse_args()

    # Load configs
    configs = load_configs()
    sim_cfg = configs["sim"]
    train_cfg = configs["train"]

    # Determine timesteps to train
    total_timesteps = args.timesteps if args.timesteps is not None else train_cfg["total_timesteps"]

    sim_manager = None
    try:
        # --- 1) Launch BeamNG and set up scenario ---
        print("--- LAUNCHING SIMULATOR FOR RESUME ---")
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch()
        sim_manager.setup_scenario(
            vehicle_model=sim_cfg["base_vehicle_model"],
            vehicle_config=sim_cfg.get("base_vehicle_config", None)
        )

        # --- 2) Create environment ---
        env = MimicEnv(sim_manager=sim_manager, config=configs)

        # --- 3) Determine checkpoint to load ---
        checkpoints_dir = os.path.join(sim_cfg["model_save_path"], "checkpoints")
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint(checkpoints_dir)
            if ckpt_path:
                print(f"Resuming from latest checkpoint: {ckpt_path}")
            else:
                # Fallback to final model if present
                final_path = os.path.join(sim_cfg["model_save_path"], "ppo_carl_final.zip")
                if os.path.isfile(final_path):
                    ckpt_path = final_path
                    print(f"No checkpoints found. Resuming from final model: {ckpt_path}")
                else:
                    raise FileNotFoundError(
                        f"No checkpoints found in: {checkpoints_dir} and no final model at: {final_path}"
                    )

        # --- 4) Load PPO model and configure logging/checkpointing ---
        model = PPO.load(ckpt_path, env=env, device="auto")

        # Reconfigure logger to use TensorBoard (like initial training)
        log_dir = sim_cfg["log_path"]
        os.makedirs(log_dir, exist_ok=True)
        logger = configure(log_dir, ["stdout", "tensorboard"])
        model.set_logger(logger)

        checkpoint_callback = CheckpointCallback(
            save_freq=20000,
            save_path=checkpoints_dir,
            name_prefix="ppo_carl",
        )

        print("\n" + "=" * 50)
        print("--- RESUMING TRAINING ---")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Timesteps (additional): {total_timesteps}")
        print(f"TensorBoard log dir: {log_dir} (run: {args.tb_name})")
        print(f"Checkpoints dir: {checkpoints_dir}")
        print("=" * 50 + "\n")

        # --- 5) Continue training (do not reset timesteps) ---
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=args.tb_name,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # --- 6) Save resumed model ---
        final_model_path = os.path.join(sim_cfg["model_save_path"], args.save_name)
        model.save(final_model_path)
        print(f"\nSaved resumed model to: {final_model_path}.zip")

    finally:
        if sim_manager:
            sim_manager.close()
            print("Simulation closed.")


if __name__ == "__main__":
    main()