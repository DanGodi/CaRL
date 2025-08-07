import time
import numpy as np
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.utils.config_loader import load_configs

def main():
    """
    Launches the simulator and performs a sequence of suspension tests:
    1. Sets initial low ride height.
    2. Smoothly raises the ride height over 30 seconds.
    3. Sets an asymmetrical stiffness profile (hard front, soft rear) at max height.
    """
    configs = load_configs()
    sim_cfg = configs['sim']
    sim_manager = None

    try:
        # --- 1. Setup the Simulation ---
        print("--- LAUNCHING SIMULATOR FOR ADVANCED SUSPENSION TEST ---")
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch()

        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['base_vehicle_model'],
            vehicle_config=sim_cfg.get('base_vehicle_config', None)
        )

        sim_manager.bng.resume()
        print("Simulation is running.")

        # --- 2. The Test Sequence ---
        print("\n" + "="*50)
        print("--- STARTING SUSPENSION SEQUENCE TEST ---")
        print("Press Ctrl+C in this terminal to stop the test.")
        print("="*50 + "\n")

        # --- Step A: Set to initial LOW position ---
        print("Step A: Setting suspension to MINIMUM height and MEDIUM stiffness...")
        params = {
            'carl_front_height_factor': 0.0,
            'carl_rear_height_factor': 0.0,
            'carl_front_stiffness_factor': 0.5,
            'carl_rear_stiffness_factor': 0.5
        }
        sim_manager.apply_vehicle_controls(params)
        time.sleep(3) # Wait 3 seconds to observe

        # --- Step B: Smoothly raise to MAXIMUM height ---
        print("\nStep B: Smoothly raising suspension to MAXIMUM height over 30 seconds...")
        num_steps = 30
        duration = 10.0
        for i in range(num_steps + 1):
            # Calculate the current height factor (from 0.0 to 1.0)
            current_height = i / num_steps
            print(f"  - Interpolation step {i}/{num_steps}: Height factor = {current_height:.2f}")

            params = {
                'carl_front_height_factor': current_height,
                'carl_rear_height_factor': current_height,
                'carl_front_stiffness_factor': 0.5,
                'carl_rear_stiffness_factor': 0.5
            }
            sim_manager.apply_vehicle_controls(params)
            time.sleep(duration / num_steps) # Wait for the correct interval

        print("  - Reached maximum height.")
        time.sleep(2) # Hold at max height for 2 seconds

        # --- Step C: Set asymmetrical stiffness ---
        print("\nStep C: Setting FRONT springs to HARD, REAR springs to SOFT at max height...")
        params = {
            'carl_front_height_factor': 1.0, # Keep height at max
            'carl_rear_height_factor': 1.0,  # Keep height at max
            'carl_front_stiffness_factor': 1.0, # 1.0 = Max stiffness (hard)
            'carl_rear_stiffness_factor': 1.0  # 0.0 = Min stiffness (soft)
        }
        sim_manager.apply_vehicle_controls(params)

        print("\nFinal state reached. Car is now at max height with hard front / soft rear suspension.")
        print("You can now switch to the BeamNG window to observe the effect.")
        print("Press Ctrl+C to end the test.")

        # Hold the final state until the user stops the script
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        # --- 3. Cleanup ---
        if sim_manager:
            sim_manager.close()
            print("Simulation closed.")

if __name__ == '__main__':
    main()