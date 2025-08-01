# carl/rl/mimic_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.beamng_control.telemetry_streamer import TelemetryStreamer
from .reward_functions import calculate_mimic_reward

class MimicEnv(gym.Env):
    """
    A Gymnasium environment for training an agent to mimic a target vehicle's telemetry.

    This version uses the "AI Driver / Target Speed" model. It uses the BeamNG AI
    to drive the base car along a pre-recorded path while attempting to match a
    pre-recorded speed profile. The RL agent's task is strictly to tune the
    chassis parameters to match the target's dynamic response.
    """
    metadata = {'render_modes': []}

    def __init__(self, sim_manager: SimulationManager, config: dict):
        super().__init__()
        
        self.sim_manager = sim_manager
        self.config = config['env']
        self.sim_config = config['sim']
        
        # Load the entire target data trace
        try:
            self.target_df = pd.read_csv(self.sim_config['target_data_path'])
        except FileNotFoundError:
            print(f"FATAL ERROR: Target data file not found at {self.sim_config['target_data_path']}")
            raise

        # --- Data Loading for Trajectory Following and Observation ---
        self.target_telemetry_full = self.target_df[self.config['observation_keys']].to_numpy()
        
        # Create a path for the AI to follow from the recorded x, y, z coordinates
        self.target_path = [{'x': row['x'], 'y': row['y'], 'z': row['z'], 't':1.0} for index, row in self.target_df.iterrows()]
        
        self.current_step_index = 0
        
        # --- Action Space (The agent's controls) ---
        self.action_param_map = self.config['action_parameter_mappings']
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_param_map),), dtype=np.float32
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # --- Observation Space (What the agent "sees") ---
        num_obs_keys = len(self.config['observation_keys'])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(num_obs_keys + self.action_space.shape[0],), 
            dtype=np.float32
        )
        
        self.telemetry_streamer = TelemetryStreamer(
            self.sim_manager.base_vehicle, 
            self.config['observation_keys']
        )
        print("MimicEnv initialized using AI Driver / Target Speed model.")

    def _scale_action(self, action: np.ndarray) -> dict:
        """Converts the normalized action from the agent [-1, 1] to physical vehicle parameters."""
        scaled_params = {}
        for i, (key, p_map) in enumerate(self.action_param_map.items()):
            scaled_value = np.interp(action[i], [-1, 1], [p_map['min'], p_map['max']])
            scaled_params[key] = scaled_value
        return scaled_params

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation vector from current state and target state."""
        current_telemetry_dict = self.telemetry_streamer.get_state()
        current_telemetry_vec = np.array([current_telemetry_dict.get(key, 0) for key in self.config['observation_keys']])
        
        target_telemetry_vec = self.target_telemetry_full[self.current_step_index]
        
        error_vector = target_telemetry_vec - current_telemetry_vec
        
        return np.concatenate([error_vector, self.last_action]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of the trajectory."""
        super().reset(seed=seed)

        self.current_step_index = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Reset the vehicle's physics and teleport to the start of the path
        self.sim_manager.reset_vehicle_physics(self.sim_manager.base_vehicle)
        start_pos = (self.target_path[0]['x'], self.target_path[0]['y'], self.target_path[0]['z'])
        self.sim_manager.base_vehicle.teleport(start_pos)
        
        # Tell the AI to start following the pre-recorded path
        self.sim_manager.base_vehicle.ai.set_mode('script', looping=False)
        self.sim_manager.base_vehicle.ai.set_script(self.target_path)
        
        self.sim_manager.bng.pause()
        
        initial_action = np.zeros(self.action_space.shape)
        initial_params = self._scale_action(initial_action)
        self.sim_manager.apply_vehicle_controls(initial_params)
        
        self.sim_manager.bng.step(5)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: np.ndarray):
        """Execute one time step within the environment."""
        # 1. Apply the agent's chassis tuning action to the vehicle
        scaled_params = self._scale_action(action)
        self.sim_manager.apply_vehicle_controls(scaled_params)
        
        # 2. Tell the AI Driver to continue along its path
        # The AI is already following the script set in reset().
        # We can optionally command a target speed here.
        target_speed_ms = self.target_df.iloc[self.current_step_index]['wheel_speed']
        self.sim_manager.base_vehicle.ai.set_speed(target_speed_ms, 'aggressive')
        
        # 3. Step the simulation forward in time.
        self.sim_manager.bng.unpause()
        # The number of physics steps should roughly match the data logging frequency
        self.sim_manager.bng.step(10) 
        self.sim_manager.bng.pause()

        self.current_step_index += 1
        
        # 4. Get the resulting state as the new observation.
        observation = self._get_observation()
        
        # 5. Calculate the reward.
        current_state = self.telemetry_streamer.get_state()
        target_state = self.target_df.iloc[self.current_step_index].to_dict()
        reward = calculate_mimic_reward(
            current_state, target_state, self.last_action, action, self.config['reward_weights']
        )
        
        # 6. Check for termination conditions.
        terminated = False
        if self.current_step_index >= len(self.target_df) - 2:
            terminated = True

        damage_data = self.sim_manager.base_vehicle.sensors.poll().get('damage', {})
        if damage_data.get('damage', 0) > self.config['damage_threshold']:
            terminated = True
            reward -= 500
        
        self.last_action = action
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Cleanly closes the environment and simulation connection."""
        print("Closing MimicEnv and simulation connection.")
        self.telemetry_streamer.close()
        self.sim_manager.close()