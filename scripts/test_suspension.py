# scripts/test_suspension.py
# A simple script to test the continuous control of the active suspension.

import time
from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.utils.config_loader import load_configs

def main():
    """
    Launches the simulator, spawns the active vehicle, and cycles the
    suspension ride height between its minimum and maximum settings.
    """
    configs = load_configs()
    sim_cfg = configs['sim']
    sim_manager = None
    
    try:
        # --- 1. Setup the Simulation ---
        print("--- LAUNCHING SIMULATOR FOR SUSPENSION TEST ---")
        sim_manager = SimulationManager(sim_cfg)
        sim_manager.launch()

        # Load the base car with its active parts configuration
        sim_manager.setup_scenario(
            vehicle_model=sim_cfg['base_vehicle_model'],
            vehicle_config=sim_cfg['base_vehicle_config']
        )
        
        # We need to unpause the simulation for the hydros to work
        sim_manager.bng.resume()
        print("Simulation is running.")

        # --- 2. The Main Control Loop ---
        print("\n" + "="*50)
        print("--- STARTING SUSPENSION CYCLE TEST ---")
        print("The car will now cycle between high and low ride height.")
        print("Press Ctrl+C in this terminal to stop the test.")
        print("="*50 + "\n")
        
        while True:
            # --- GO TO MAXIMUM HEIGHT ---
            print("Setting suspension to MAXIMUM height (factor = 1.0)")
            # We send a dictionary with the electrics value our LUA expects
            params = {'carl_height_factor': 1.0}
            sim_manager.apply_vehicle_controls(params)
            time.sleep(2) # Wait 2 seconds

            # --- GO TO MINIMUM HEIGHT ---
            print("Setting suspension to MINIMUM height (factor = 0.0)")
            params = {'carl_height_factor': 0.0}
            sim_manager.apply_vehicle_controls(params)
            time.sleep(2) # Wait 2 seconds

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        # --- 3. Cleanup ---
        if sim_manager:
            sim_manager.close()
            print("Simulation closed.")

if __name__ == '__main__':
    main()