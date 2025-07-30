# scripts/01_generate_target_data.py
# --- FINAL AUTOMATED VERSION ---

import time
import pandas as pd
from pathlib import Path
import keyboard

from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.telemetry_streamer import TelemetryStreamer

def generate_target_data(config: dict):
    """
    Generates target data in a fully automated way.
    1. Launches a new BeamNG.tech instance.
    2. Spawns the target car on the specified map.
    3. Gives control to the user to drive and record a path.
    4. Automatically replays the path with AI to generate a clean data log.
    """
    sim_manager = None
    try:
        # --- Part 1: Launch Sim and Setup for Manual Driving ---
        sim_manager = SimulationManager(config['sim'])
        sim_manager.launch() # Use launch() to create a dedicated instance

        # Setup the scenario with the target vehicle model
        # The setup_scenario method will handle map loading and vehicle spawning
        sim_manager.setup_scenario(spawn_target=False)
        
        # We use the 'base_vehicle' spawned by the manager as our target for this script
        player_vehicle = sim_manager.base_vehicle
        # Make sure it's the correct model from the config
        player_vehicle.model = config['sim']['target_vehicle_model']

        print("\n" + "="*50)
        print("--- READY FOR PATH RECORDING ---")
        print("Simulator is ready. The BeamNG window is now active.")
        print(f"Drive the '{player_vehicle.model}' on the '{config['sim']['map']}' map.")
        print("\nPress 'ESC' key to stop recording.")
        print("="*50 + "\n")

        script = []
        player_vehicle.ai.set_mode('manual')
        
        # Loop to record path nodes until user presses ESC
        # Using the 'keyboard' library is more reliable than Ctrl+C in this context
        while not keyboard.is_pressed('esc'):
            player_vehicle.sensors.poll()
            pos = player_vehicle.state['pos']
            script.append({'x': pos[0], 'y': pos[1], 'z': pos[2], 't': 1.0})
            time.sleep(0.2)
        
        print(f"\nRecording stopped. Recorded {len(script)} nodes.")

        # --- Part 2: Replay the path with AI and log telemetry ---
        print("\n--- Telemetry Logging Phase ---")
        print("Setting up AI replay using the same vehicle...")

        target_vehicle = player_vehicle

        print("Teleporting to start and setting AI script...")
        start_pos = (script[0]['x'], script[0]['y'], script[0]['z'])
        target_vehicle.teleport(start_pos, reset=True)
        
        target_vehicle.ai.set_mode('script', looping=False)
        target_vehicle.ai.set_script(script)

        obs_keys = config['env']['observation_keys']
        position_keys = ['time', 'x', 'y', 'z', 'wheel_speed']
        all_keys_to_log = list(set(obs_keys + position_keys))
        
        streamer = TelemetryStreamer(target_vehicle, all_keys_to_log)

        sim_manager.bng.unpause()
        telemetry_log = []
        start_time = time.time()
        
        print("Logging data... This will end automatically when the path is complete.")
        while len(telemetry_log) == 0 or target_vehicle.ai.is_script_done() is False:
            sim_manager.bng.poll_sensors_and_state()
            
            state = streamer.get_state()
            pos = target_vehicle.state['pos']
            
            state['time'] = time.time() - start_time
            state['x'], state['y'], state['z'] = pos[0], pos[1], pos[2]
            
            telemetry_log.append(state)
            time.sleep(1 / 50)

        print(f"Logged {len(telemetry_log)} telemetry points.")
        
        output_path = Path(config['sim']['target_data_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(telemetry_log)
        df.to_csv(output_path, index=False)
        print(f"Target telemetry saved to: {output_path}")

    finally:
        if sim_manager:
            sim_manager.close()

if __name__ == '__main__':
    # Add a new dependency for keyboard input
    # In your terminal (with venv active): pip install keyboard
    configs = load_configs()
    generate_target_data(configs)