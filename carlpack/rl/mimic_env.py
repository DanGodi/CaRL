import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

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
        self.reset_time = 0.0
        
        self.target_df = pd.read_csv(self.sim_config['target_data_path'])
        self.target_telemetry_full = self.target_df[self.config['observation_keys']].to_numpy()
        self.max_target_time = self.target_df['time'].iloc[-1]

        self.normalization_values = {}
        # We only need to calculate max values for the keys used in the reward function.
        reward_keys = self.config.get('reward_weights', {}).keys()
        for key in reward_keys:
            if key in self.target_df.columns:
                # Find the maximum absolute value for the column.
                max_abs_val = self.target_df[key].abs().max()
                # Store it, ensuring it's not zero to prevent division errors.
                self.normalization_values[key] = max_abs_val if max_abs_val > 1e-6 else 1.0
        
        
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
        self.sim_manager.base_vehicle.ai.set_script(self.target_path_with_time)
        self._resume_pending = True
        self.sim_manager.base_vehicle.sensors.poll()
        current_state = self.telemetry_streamer.get_state()
        self.reset_time = current_state['time']
        
        observation = self._get_observation()
        info = {}
        self.sim_manager.bng.resume()
        return observation, info

    def step(self, action: np.ndarray):
        scaled_params = self._scale_action(action)
        if self._resume_pending:
            self.sim_manager.apply_vehicle_controls(scaled_params)
            self.sim_manager.bng.resume()
            self._resume_pending = False
        else:
            self.sim_manager.apply_vehicle_controls(scaled_params)

        # Poll sensors BEFORE stepping the simulation
        self.sim_manager.base_vehicle.sensors.poll()
        current_state = self.telemetry_streamer.get_state()
        elapsed_episode_time = current_state['time'] - self.reset_time
        current_y = current_state['y']
        insertion_point = self.target_df['y'].searchsorted(current_y, side='right')
        self.current_step_index = max(0, insertion_point)
        observation = self._get_observation()

        terminated = elapsed_episode_time >= self.max_target_time + 1

        target_state = self.target_df.iloc[min(self.current_step_index, len(self.target_df) - 1)].to_dict()
        if np.random.rand() < 1/1000:
            print(current_y, target_state)

        reward = calculate_mimic_reward(
            current_state, target_state, self.last_action, action, self.config['reward_weights'], self.normalization_values
        )

        self.last_action = action
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def close(self):
        # ... (no changes needed here)
        print("Closing MimicEnv and simulation connection.")
        self.telemetry_streamer.close()
        self.sim_manager.close()