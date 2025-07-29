# scripts/03_evaluate_agent.py
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.rl.mimic_env import MimicEnv

def evaluate_agent(model_path: str, output_csv_name: str):
    """
    Loads a trained agent and runs it in the environment to produce an
    evaluation telemetry log.
    """
    configs = load_configs()
    sim_cfg = configs['sim']
    
    sim_manager = None
    try:
        # Setup simulation and environment
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.connect()
        # Spawn both vehicles for visual comparison
        sim_manager.setup_scenario(spawn_target=True) 

        env = MimicEnv(sim_manager=sim_manager, config=configs)
        
        # Load the trained model
        model = PPO.load(model_path, env=env)
        print(f"Loaded model from: {model_path}")
        
        # Run the evaluation loop
        obs, _ = env.reset()
        done = False
        mimic_telemetry_log = []
        
        print("\n--- STARTING EVALUATION ---")
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Log telemetry for the mimic car
            state = env.telemetry_streamer.get_state()
            pos = env.sim_manager.base_vehicle.state['pos']
            state['x'], state['y'], state['z'] = pos[0], pos[1], pos[2]
            mimic_telemetry_log.append(state)
        
        # Save the results
        output_path = Path(sim_cfg['data_path']) / "evaluation_results" / output_csv_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(mimic_telemetry_log)
        df.to_csv(output_path, index=False)

        print("\n--- EVALUATION COMPLETE ---")
        print(f"Evaluation results saved to: {output_path}")
        print(f"You can now analyze this file against '{sim_cfg['target_data_path']}'")
        
    finally:
        if sim_manager:
            sim_manager.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained Chimera agent.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model .zip file.")
    parser.add_argument('--output_name', type=str, default='evaluation_results.csv', help="Name of the output CSV file.")
    
    args = parser.parse_args()
    evaluate_agent(args.model_path, args.output_name)