import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import time

from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.beamng_control.telemetry_streamer import TelemetryStreamer
from .reward_functions import calculate_mimic_reward

class MimicEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, sim_manager: SimulationManager, config: dict):
        super().__init__()
        
        self.sim_manager = sim_manager
        self.config = config['env']
        self.sim_config = config['sim']
        
        self.target_df = pd.read_csv(self.sim_config['target_data_path'])
        self.target_telemetry_full = self.target_df[self.config['observation_keys']].to_numpy()
        
        self.target_path_with_time = [
            {'x': row['x'], 'y': row['y'], 'z': row['z'], 't': row['time']} 
            for _, row in self.target_df.iterrows()
        ]
        
        self.current_step_index = 0
        
        self.action_param_map = self.config['action_parameter_mappings']
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.action_param_map),), dtype=np.float32)
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        num_obs_keys = len(self.config['observation_keys'])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(num_obs_keys + self.action_space.shape[0],), 
            dtype=np.float32
        )
        
        self.telemetry_streamer = TelemetryStreamer(self.sim_manager.base_vehicle, self.config['observation_keys'], bng=self.sim_manager.bng)
        print("MimicEnv initialized.")

    def _scale_action(self, action: np.ndarray) -> dict:
        scaled_params = {}
        for i, (key, p_map) in enumerate(self.action_param_map.items()):
            scaled_value = np.interp(action[i], [-1, 1], [p_map['min'], p_map['max']])
            scaled_params[key] = scaled_value
        return scaled_params

    def _get_observation(self) -> np.ndarray:
        current_telemetry_dict = self.telemetry_streamer.get_state()
        current_telemetry_vec = np.array([current_telemetry_dict.get(key, 0) for key in self.config['observation_keys']])
        idx = min(self.current_step_index, len(self.target_telemetry_full) - 1)
        target_telemetry_vec = self.target_telemetry_full[idx]
        error_vector = target_telemetry_vec - current_telemetry_vec
        return np.concatenate([error_vector, self.last_action]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_index = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        self.sim_manager.bng.pause()
        self.sim_manager.base_vehicle.ai.set_mode('disabled')
        start_pos = (self.target_path_with_time[0]['x'], self.target_path_with_time[0]['y'], self.target_path_with_time[0]['z'])
        self.sim_manager.base_vehicle.teleport(start_pos,(0, 0, 1, 0), reset=True)
        self.sim_manager.base_vehicle.sensors.poll()
        self.sim_manager.base_vehicle.ai.set_mode('script')
        self.sim_manager.base_vehicle.ai.set_script(self.target_path_with_time)
        initial_action = np.zeros(self.action_space.shape)
        initial_params = self._scale_action(initial_action)
        self.sim_manager.apply_vehicle_controls(initial_params)
        
        # We start the episode paused, ready for the first step() call.
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        # --- THE STABLE, DETERMINISTIC LOOP ---
        scaled_params = self._scale_action(action)
        self.sim_manager.apply_vehicle_controls(scaled_params)

        # Poll sensors BEFORE stepping the simulation
        self.sim_manager.base_vehicle.sensors.poll()
        observation = self._get_observation()

        # We now increment our step index manually. This is our reliable "clock".
        self.current_step_index += 1

        terminated = False
        if self.current_step_index >= len(self.target_df) - 2:
            terminated = True

        current_state = self.telemetry_streamer.get_state()
        target_state = self.target_df.iloc[min(self.current_step_index, len(self.target_df) - 1)].to_dict()
        reward = calculate_mimic_reward(
            current_state, target_state, self.last_action, action, self.config['reward_weights']
        )

        if 'damage' in self.sim_manager.base_vehicle.sensors._sensors:
            damage_data = self.sim_manager.base_vehicle.sensors['damage']
        else:
            damage_data = {}
        if damage_data.get('damage', 0) > self.config.get('damage_threshold', 1000):
            terminated = True
            reward -= 500

        self.last_action = action
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def close(self):
        # ... (no changes needed here)
        print("Closing MimicEnv and simulation connection.")
        self.telemetry_streamer.close()
        self.sim_manager.close()