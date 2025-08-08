import pandas as pd
from pathlib import Path
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.rl.mimic_env import MimicEnv

class TrainingLoggerCallback(BaseCallback):
    """
    A custom callback to log detailed telemetry during the training loop.
    It activates logging for a specific episode and stops training after a set number of episodes.
    """
    def __init__(self, log_episode: int, stop_after_n_episodes: int, output_csv_name: str, verbose=0):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.log_episode = log_episode
        self.stop_after_n_episodes = stop_after_n_episodes
        self.output_csv_name = output_csv_name
        self.episode_count = 0
        self.is_logging_active = False
        self.realtime_log = []

    def _on_rollout_start(self) -> None:
        """Called at the start of each new rollout (effectively, each episode)."""
        if self.episode_count == self.log_episode:
            if self.verbose > 0:
                print(f"\n--- CALLBACK: Starting logging for Episode {self.episode_count + 1} ---")
            self.is_logging_active = True
        else:
            if self.verbose > 0:
                print(f"\n--- CALLBACK: Starting Episode {self.episode_count + 1} (logging disabled) ---")
            self.is_logging_active = False

    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        Returns False to stop training.
        """
        # Stop training if the desired number of episodes has been completed.
        if self.episode_count >= self.stop_after_n_episodes:
            if self.verbose > 0:
                print(f"--- CALLBACK: Reached {self.episode_count} episodes. Stopping training. ---")
            return False

        if self.is_logging_active:
            # Access the underlying MimicEnv instance from the vectorized environment
            env = self.training_env.envs[0]
            
            # Get the last action from the model's local variables, which is available after the step
            action = self.locals['actions'][0]

            # Log the detailed data for the current step
            current_telemetry = env.telemetry_streamer.get_state()
            target_idx = min(env.current_step_index, len(env.target_df) - 1)
            target_telemetry = env.target_df.iloc[target_idx].to_dict()

            log_entry = {}
            for key, val in current_telemetry.items():
                log_entry[f'mimic_{key}'] = val
            for key, val in target_telemetry.items():
                log_entry[f'target_{key}'] = val
            
            for i, act_val in enumerate(action):
                log_entry[f'action_{i}'] = act_val
            
            self.realtime_log.append(log_entry)
        
        return True # Continue training

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if self.is_logging_active:
            if self.verbose > 0:
                print(f"--- CALLBACK: Finished logging for Episode {self.episode_count + 1} ---")
            self.is_logging_active = False # Stop logging after the target episode is done
        self.episode_count += 1

    def _on_training_end(self) -> None:
        """Called at the very end of the training process."""
        if self.realtime_log:
            output_path = Path(self.output_csv_name)
            df = pd.DataFrame(self.realtime_log)
            df.to_csv(output_path, index=False)
            if self.verbose > 0:
                print(f"\n--- CALLBACK: Debug log saved to: {output_path.absolute()} ---")

def debug_training_session(output_csv_name: str):
    """
    Runs the PPO training loop and uses a callback to log detailed data
    from the third episode, stopping after exactly three episodes.
    """
    configs = load_configs()
    sim_cfg = configs['sim']
    train_cfg = configs['train']
    
    sim_manager = None
    try:
        # --- 1. Setup Simulation and Environment ---
        print("--- LAUNCHING SIMULATOR FOR DEBUGGING TRAINING ---")
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch()
        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['base_vehicle_model'],
            vehicle_config=sim_cfg.get('base_vehicle_config', None)
        )
        
        # The environment must be wrapped for SB3.
        # A seed is provided to prevent a NumPy integer overflow error on some systems.
        env = make_vec_env(MimicEnv, n_envs=1, env_kwargs=dict(sim_manager=sim_manager, config=configs), seed=42)

        # --- 2. Setup Model and Callback ---
        model = PPO(
            env=env,
            verbose=1,
            tensorboard_log=None, # Disable tensorboard for this debug run
            **train_cfg['ppo_params']
        )

        # Instantiate the callback to log the 3rd episode (index 2) and stop after 3 total episodes.
        logger_callback = TrainingLoggerCallback(
            log_episode=2, 
            stop_after_n_episodes=3, 
            output_csv_name=output_csv_name, 
            verbose=1
        )

        # --- 3. Run model.learn() with the Callback ---
        print("\n--- STARTING TRAINING LOOP WITH LOGGER (WILL STOP AFTER 3 EPISODES) ---")
        # Run for a large number of timesteps; the callback will handle stopping the training.
        model.learn(
            total_timesteps=1_000_000,
            callback=logger_callback
        )

    finally:
        if sim_manager:
            print("Closing simulation.")
            sim_manager.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug the CaRL training loop by logging the 3rd episode.")
    parser.add_argument('--output_name', type=str, default='debug_training_loop_log.csv', help="Name for the output CSV file.")

    args = parser.parse_args()
    debug_training_session(args.output_name)