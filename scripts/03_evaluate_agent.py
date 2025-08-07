# scripts/03_evaluate_agent.py
# --- FINAL, CORRECTED VERSION ---

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
        # --- 1. Setup simulation and environment ---
        print("--- LAUNCHING SIMULATOR FOR EVALUATION ---")
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch() # Use launch() for a clean, automated start

        # Spawn BOTH the base car and a "ghost" target car for visual comparison
        # NOTE: This requires a more advanced setup_scenario, let's keep it simple for now
        # and just spawn the base car that the agent will control.
        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['base_vehicle_model'],
            vehicle_config=sim_cfg.get('base_vehicle_config', None),
            spawn_target=False # Keep this as False for now
        )

        # The environment is created just like in training
        env = MimicEnv(sim_manager=sim_manager, config=configs)
        
        # --- 2. Load the Trained Model ---
        # The environment must be passed to load() so the model knows the
        # action and observation spaces.
        model = PPO.load(model_path, env=env)
        print(f"Successfully loaded model from: {model_path}")
        
        # --- 3. Run the Evaluation Loop ---
        obs, _ = env.reset()
        done = False
        mimic_telemetry_log = []
        
        print("\n--- STARTING EVALUATION RUN ---")
        while not done:
            # Use deterministic=True to get the agent's best action without random exploration
            action, _states = model.predict(obs, deterministic=True)
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Log the telemetry and position for this step
            # We get this directly from the info dictionary returned by our modified env.
            # This is more robust than polling again.
            
            # Let's poll manually for now to be safe
            env.sim_manager.base_vehicle.sensors.poll()
            state = env.telemetry_streamer.get_state()
            pos = env.sim_manager.base_vehicle.state.get('pos', [0,0,0])

            state['x'] = pos[0]
            state['y'] = pos[1]
            state['z'] = pos[2]
            
            mimic_telemetry_log.append(state)

        # --- 4. Save the Results ---
        # Construct the full path for the output file
        output_dir = Path(sim_cfg['data_path']) / "evaluation_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_csv_name
        
        df = pd.DataFrame(mimic_telemetry_log)
        
        # Re-order columns to match the target data file for easier comparison
        try:
            target_df = pd.read_csv(sim_cfg['target_data_path'])
            # Use the column order from the target file if it exists
            df = df.reindex(columns=target_df.columns, fill_value=0)
        except Exception as e:
            print(f"Could not re-order columns based on target file: {e}")

        df.to_csv(output_path, index=False)

        print("\n--- EVALUATION COMPLETE ---")
        print(f"Evaluation results saved to: {output_path}")
        print(f"You can now analyze this file using the plot_evaluation.py script.")
        
    finally:
        if sim_manager:
            sim_manager.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained CaRL agent.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model .zip file.")
    parser.add_argument('--output_name', type=str, default='evaluation_results.csv', help="Name for the output CSV file.")
    
    args = parser.parse_args()
    evaluate_agent(args.model_path, args.output_name)