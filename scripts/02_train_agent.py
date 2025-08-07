import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.rl.mimic_env import MimicEnv

def main():
    configs = load_configs()
    sim_cfg = configs['sim']
    train_cfg = configs['train']

    sim_manager = None
    try:
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch()

        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['base_vehicle_model'],
            vehicle_config=sim_cfg['base_vehicle_config']
        )

        env = MimicEnv(sim_manager=sim_manager, config=configs)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=20000,
            save_path=os.path.join(sim_cfg['model_save_path'], 'checkpoints'),
            name_prefix='ppo_carl'
        )

        model = PPO(
            env=env,
            tensorboard_log=sim_cfg['log_path'],
            verbose=1,
            **train_cfg['ppo_params']
        )

        print("\n" + "="*50)
        print("--- STARTING REINFORCEMENT LEARNING TRAINING ---")
        print(f"Logging to: {sim_cfg['log_path']}")
        print(f"Models will be saved in: {sim_cfg['model_save_path']}")
        print("="*50 + "\n")

        model.learn(
            total_timesteps=train_cfg['total_timesteps'],
            callback=checkpoint_callback,
            tb_log_name="PPO_CaRL_Run"
        )
        
        final_model_path = os.path.join(sim_cfg['model_save_path'], 'ppo_carl_final')
        model.save(final_model_path)

        print("\n" + "="*50)
        print("--- TRAINING COMPLETE ---")
        print(f"Final model saved to: {final_model_path}.zip")
        print("="*50 + "\n")

    finally:
        if sim_manager:
            sim_manager.close()

if __name__ == '__main__':
    main()