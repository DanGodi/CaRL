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

    This version uses the "Human Input Replay" method. It replays recorded
    steering, throttle, and brake inputs, while the RL agent's task is
    strictly to tune the chassis parameters to match the target's response.
    """
    metadata = {'render_modes': []}

    def __init__(self, sim_manager: SimulationManager, config: dict):
        super().__init__()
        
        self.sim_manager = sim_manager
        self.config = config['env']
        self.sim_config = config['sim']
        
        # Load the entire target telemetry trace from the pre-generated CSV
        try:
            self.target_df = pd.read_csv(self.sim_config['target_data_path'])
        except FileNotFoundError:
            print(f"FATAL ERROR: Target data file not found at {self.sim_config['target_data_path']}")
            print("Please run scripts/01_generate_target_data.py first.")
            raise

        # --- Data Loading for Replay and Observation ---
        self.target_telemetry_full = self.target_df[self.config['observation_keys']].to_numpy()
        
        # Load the driver input data for replay.
        # This is the core of the "Human Input Replay" method.
        try:
            self.driver_inputs = self.target_df[['steering_input', 'throttle_input', 'brake_input']].to_numpy()
        except KeyError:
            print("FATAL ERROR: The target CSV is missing required input columns.")
            print("Required columns: 'steering_input', 'throttle_input', 'brake_input'.")
            print("Please re-run the data generation script with the updated version.")
            raise
            
        self.current_step_index = 0
        
        # --- Action Space (The agent's controls) ---
        # A continuous space for each active parameter (e.g., spring stiffness, toe angle).
        # Values are normalized to [-1, 1] and will be scaled to physical ranges.
        self.action_param_map = self.config['action_parameter_mappings']
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.action_param_map),), dtype=np.float32
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # --- Observation Space (What the agent "sees") ---
        # Consists of:
        # 1. The error vector (target telemetry - current telemetry).
        # 2. The agent's last applied action (to provide state information).
        num_obs_keys = len(self.config['observation_keys'])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(num_obs_keys + self.action_space.shape[0],), 
            dtype=np.float32
        )
        
        # Setup the telemetry streamer to get data from the base vehicle.
        self.telemetry_streamer = TelemetryStreamer(
            self.sim_manager.base_vehicle, 
            self.config['observation_keys']
        )
        print("MimicEnv initialized using Human Input Replay model.")

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
        current_telemetry_vec = np.array([current_telemetry_dict.get(key, 0) for key in self.config['observation_keys']])
        
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
        
        # Find the starting position from the recorded data
        start_pos = (
            self.target_df.iloc[0]['x'], 
            self.target_df.iloc[0]['y'], 
            self.target_df.iloc[0]['z']
        )
        self.sim_manager.base_vehicle.teleport(start_pos)
        
        # Pausing is crucial for getting a stable initial observation
        self.sim_manager.bng.pause()
        
        # Apply a default (neutral) chassis action
        initial_action = np.zeros(self.action_space.shape)
        initial_params = self._scale_action(initial_action)
        self.sim_manager.apply_vehicle_controls(initial_params)
        
        # Step the simulation a few times to let systems settle
        self.sim_manager.bng.step(5)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: np.ndarray):
        """Execute one time step within the environment."""
        # 1. Apply the agent's chassis tuning action to the vehicle
        scaled_params = self._scale_action(action)
        self.sim_manager.apply_vehicle_controls(scaled_params)
        
        # 2. Replay the recorded driver inputs for this timestep.
        # This is the core of this method: the agent does not control the car's driving.
        if self.current_step_index < len(self.driver_inputs):
            inputs = self.driver_inputs[self.current_step_index]
            self.sim_manager.base_vehicle.control(
                steering=inputs[0],
                throttle=inputs[1],
                brake=inputs[2]
            )
        
        # 3. Step the simulation forward in time.
        self.sim_manager.bng.unpause()
        # The number of physics steps might need tuning. It should correspond to the
        # logging frequency in the data generation script.
        # Example: 50Hz logging -> 20ms/step. BeamNG physics is ~2ms/step. So ~10 sim steps.
        self.sim_manager.bng.step(10) 
        self.sim_manager.bng.pause()

        self.current_step_index += 1
        
        # 4. Get the resulting state as the new observation.
        observation = self._get_observation()
        
        # 5. Calculate the reward based on how well the mimicry worked.
        current_state = self.telemetry_streamer.get_state()
        target_state = self.target_df.iloc[self.current_step_index].to_dict()
        reward = calculate_mimic_reward(
            current_state, target_state, self.last_action, action, self.config['reward_weights']
        )
        
        # 6. Check for termination conditions.
        terminated = False
        if self.current_step_index >= len(self.target_df) - 2:
            terminated = True
            print("Reached end of trajectory.")

        damage_data = self.sim_manager.base_vehicle.sensors.poll().get('damage', {})
        if damage_data.get('damage', 0) > self.config['damage_threshold']:
            terminated = True
            reward -= 500 # Apply a large penalty for crashing
            print(f"Termination: Vehicle damage exceeded threshold ({damage_data.get('damage', 0)}).")
        
        self.last_action = action
        truncated = False # Use 'terminated' for end-of-episode conditions.
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Cleanly closes the environment and simulation connection."""
        print("Closing MimicEnv and simulation connection.")
        self.telemetry_streamer.close()
        self.sim_manager.close()