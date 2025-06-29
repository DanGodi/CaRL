# scripts/02_train_agent.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from chimera.utils.config_loader import load_configs
from chimera.beamng_control.simulation_manager import SimulationManager
from chimera.rl.mimic_env import MimicEnv

def main():
    configs = load_configs()
    sim_cfg = configs['sim']
    train_cfg = configs['train']

    sim_manager = None
    try:
        # 1. Setup the Simulation Manager and start BeamNG
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.connect()
        # Spawn only the base vehicle for training
        sim_manager.setup_scenario(spawn_target=False)

        # 2. Create the Gym Environment
        env = MimicEnv(sim_manager=sim_manager, config=configs)
        
        # Optional: Check if the custom environment is valid
        # check_env(env) 
        # print("Environment check passed.")

        # 3. Setup Callbacks
        # Save a checkpoint of the model every N steps
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=os.path.join(sim_cfg['model_save_path'], 'checkpoints'),
            name_prefix='ppo_mimic'
        )

        # 4. Setup and Train the RL Model
        model = PPO(
            env=env,
            tensorboard_log=sim_cfg['log_path'],
            verbose=1,
            **train_cfg['ppo_params']
        )

        print("\n" + "="*50)
        print("--- STARTING REINFORCEMENT LEARNING TRAINING ---")
        print(f"Algorithm: PPO")
        print(f"Total Timesteps: {train_cfg['total_timesteps']}")
        print(f"Logging to: {sim_cfg['log_path']}")
        print(f"Models will be saved in: {sim_cfg['model_save_path']}")
        print("="*50 + "\n")

        # The main training loop
        model.learn(
            total_timesteps=train_cfg['total_timesteps'],
            callback=checkpoint_callback,
            tb_log_name="PPO_Mimic_Run"
        )
        
        # Save the final model
        final_model_path = os.path.join(sim_cfg['model_save_path'], 'ppo_mimic_final')
        model.save(final_model_path)

        print("\n" + "="*50)
        print("--- TRAINING COMPLETE ---")
        print(f"Final model saved to: {final_model_path}.zip")
        print("="*50 + "\n")

    finally:
        # Ensure the simulation is closed even if an error occurs
        if sim_manager:
            sim_manager.close()

if __name__ == '__main__':
    main()