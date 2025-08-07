import numpy as np
import json
from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.rl.mimic_env import MimicEnv

def main():
    configs = load_configs()
    sim_manager = SimulationManager(configs['sim'])
    sim_manager.launch()
    sim_manager.setup_scenario(
        vehicle_model=configs['sim']['base_vehicle_model'],
        vehicle_config=configs['sim']['base_vehicle_config']
    )
    env = MimicEnv(sim_manager=sim_manager, config=configs)

    log_data = []

    for episode in range(3):
        obs, _ = env.reset()
        sim_manager.bng.resume()
        actions = []
        rewards = []
        telemetry_list = []
        done = False
        while not done:
            action = env.action_space.sample()
            step_result = env.step(action)
            # Handle both Gymnasium and Gym API
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated, truncated = done, False
            actions.append(action.tolist())
            rewards.append(reward)
            telemetry_list.append(env.telemetry_streamer.get_state())
            done = terminated or truncated

        idx = min(env.current_step_index, len(env.target_df) - 1)
        target_telemetry = env.target_df.iloc[idx].to_dict()
        log_entry = {
            "episode": episode,
            "final_current_telemetry": telemetry_list[-1] if telemetry_list else {},
            "final_target_telemetry": target_telemetry,
            "total_reward": float(np.sum(rewards)),
            "actions": actions
        }
        log_data.append(log_entry)

    # Save log to file
    with open("debug_two_episode_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    env.close()
    sim_manager.close()

if __name__ == "__main__":
    main()