# chimera/rl/reward_functions.py
import numpy as np

def calculate_mimic_reward(
    current_state: dict, 
    target_state: dict,
    last_action: np.ndarray,
    current_action: np.ndarray,
    weights: dict,
    normalization_values: dict
) -> float:
    """
    Calculates the reward for the current timestep based on mimicry error and penalties.

    Args:
        current_state: Dictionary of the mimic car's current telemetry.
        target_state: Dictionary of the target car's telemetry for this step.
        last_action: The action taken in the previous step.
        current_action: The action taken in the current step.
        weights: Dictionary of weights for each reward component.

    Returns:
        The scalar reward value.
    """
    total_reward = 0.0
    
    # --- 1. Mimicry Reward (primary objective) ---
    for key, weight in weights.items():
        if key in current_state and key in target_state:
            error = target_state[key] - current_state[key]
            
            max_val = normalization_values[key]
            normalized_error = error / max_val
            total_reward -= (normalized_error ** 2) * weight

    # --- 2. Action Penalties (to encourage smooth and efficient control) ---
    # Penalty for jerky actions
    if 'action_smoothness_penalty' in weights:
        smoothness_penalty = np.sum(np.square(current_action - last_action))
        total_reward -= smoothness_penalty * weights['action_smoothness_penalty']
        
    # Penalty for large actions
    """if 'action_magnitude_penalty' in weights:
        magnitude_penalty = np.sum(np.square(current_action))
        total_reward -= magnitude_penalty * weights['action_magnitude_penalty']"""
        
    return total_reward