# chimera/rl/mimic_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from chimera.beamng_control.simulation_manager import SimulationManager
from chimera.beamng_control.telemetry_streamer import TelemetryStreamer
from .reward_functions import calculate_mimic_reward

class MimicEnv(gym.Env):
    """
    A Gymnasium environment for training an agent to mimic a target vehicle's telemetry
    by controlling active vehicle components in BeamNG.
    """
    metadata = {'render_modes': []}

    def __init__(self, sim_manager: SimulationManager, config: dict):
        super().__init__()
        
        self.sim_manager = sim_manager
        self.config = config['env']
        self.sim_config = config['sim']
        
        # Load the entire target telemetry trace from the pre-generated CSV
        self.target_df = pd.read_csv(self.sim_config['target_data_path'])
        self.target_telemetry_full = self.target_df[self.config['observation_keys']].to_numpy()
        self.target_path = self.target_df[['x', 'y', 'z']].to_numpy()

        self.current_step_index = 0
        
        # --- Action Space ---
        # A continuous space where each element corresponds to an active parameter.
        # The values are normalized to [-1, 1] and will be scaled later.
        self.action_param_map = self.config['action_parameter_mappings']
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_param_map),), dtype=np.float32
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # --- Observation Space ---
        # Consists of:
        # 1. The error vector (target telemetry - current telemetry).
        # 2. The agent's last applied action.
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
        print("MimicEnv initialized.")

    def _scale_action(self, action: np.ndarray) -> dict:
        """Converts the normalized action from the agent [-1, 1] to physical vehicle parameters."""
        scaled_params = {}
        for i, (key, p_map) in enumerate(self.action_param_map.items()):
            # Linear interpolation from [-1, 1] to [min, max]
            scaled_value = np.interp(action[i], [-1, 1], [p_map['min'], p_map['max']])
            scaled_params[key] = scaled_value
        return scaled_params

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation vector from current state and target state."""
        current_telemetry_dict = self.telemetry_streamer.get_state()
        current_telemetry_vec = np.array([current_telemetry_dict[key] for key in self.config['observation_keys']])
        
        target_telemetry_vec = self.target_telemetry_full[self.current_step_index]
        
        error_vector = target_telemetry_vec - current_telemetry_vec
        
        # Concatenate error with the last action to form the full observation
        return np.concatenate([error_vector, self.last_action]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of the trajectory."""
        super().reset(seed=seed)

        self.current_step_index = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Reset the vehicle's physics state (position, velocity, etc.)
        self.sim_manager.reset_vehicle_physics(self.sim_manager.base_vehicle)
        
        # Teleport to the starting point of the recorded trajectory
        start_pos = tuple(self.target_path[0])
        self.sim_manager.base_vehicle.teleport(start_pos)
        
        # Pausing is crucial for getting a stable initial observation
        self.sim_manager.bng.pause()
        
        # Apply a default (neutral) action
        initial_action = np.zeros(self.action_space.shape)
        initial_params = self._scale_action(initial_action)
        self.sim_manager.apply_vehicle_controls(initial_params)
        
        # Step the simulation a few times to let systems settle
        self.sim_manager.bng.step(5)
        
        observation = self._get_observation()
        info = {} # Info dict is standard in Gymnasium
        
        return observation, info

    def step(self, action: np.ndarray):
        """Execute one time step within the environment."""
        # 1. Apply the scaled action to the vehicle
        scaled_params = self._scale_action(action)
        self.sim_manager.apply_vehicle_controls(scaled_params)
        
        # 2. Drive the car along the target path using AI
        # This decouples the "driving" (steering/throttle) from the "tuning" (active systems).
        # The agent's only job is to tune the car's response, not drive it.
        target_node = {
            'pos': tuple(self.target_path[self.current_step_index + 1]),
            'speed': 25 # I AM UNSURE HOW TO GET TARGET SPEED FROM CSV. SETTING TO A CONSTANT FOR NOW.
                       # A BETTER WAY IS TO RECORD SPEED IN THE TARGET DATA AND USE IT HERE.
        }
        self.sim_manager.base_vehicle.ai.set_script([target_node], up_dir=(0, 0, 1))

        # 3. Step the simulation
        self.sim_manager.bng.unpause()
        # Step by a larger amount to ensure the car moves between points
        # The number of steps might need tuning depending on physics step size.
        self.sim_manager.bng.step(20) 
        self.sim_manager.bng.pause()

        self.current_step_index += 1
        
        # 4. Get new observation
        observation = self._get_observation()
        
        # 5. Calculate reward
        current_state = self.telemetry_streamer.get_state()
        target_state = self.target_df.iloc[self.current_step_index].to_dict()
        reward = calculate_mimic_reward(
            current_state, target_state, self.last_action, action, self.config['reward_weights']
        )
        
        # 6. Check for termination conditions
        terminated = False
        if self.current_step_index >= len(self.target_df) - 2:
            terminated = True
            print("Reached end of trajectory.")

        # Check for crash/damage
        damage_data = self.sim_manager.base_vehicle.sensors.poll().get('damage', {})
        if damage_data.get('damage', 0) > self.config['damage_threshold']:
            terminated = True
            reward -= 500 # Large penalty for crashing
            print("Termination: Vehicle damaged.")
        
        self.last_action = action
        truncated = False # We use terminated for end-of-episode
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Cleanly closes the environment and simulation connection."""
        self.telemetry_streamer.close()
        self.sim_manager.close()