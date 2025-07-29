# scripts/01_generate_target_data.py
# --- FULLY CORRECTED AND ROBUST VERSION ---

import time
import pandas as pd
from pathlib import Path

from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.telemetry_streamer import TelemetryStreamer

def generate_target_data(config: dict):
    """
    Generates target data by first recording a path driven by the user,
    and then having the AI replay that path to generate a clean, consistent
    telemetry and speed profile log.
    """
    sim_manager = None
    try:
        # --- Part 1: Connect to a running game to find the player vehicle ---
        print("--- Path Recording Phase ---")
        print("Please connect to a running instance of BeamNG.tech.")
        print("Drive the target vehicle along your desired path.")
        print("Press Ctrl+C in this console when you are finished driving.")

        sim_manager = SimulationManager(config['sim'])
        sim_manager.connect()
        
        # --- The Correct Vehicle Discovery Process ---
        print("Getting active scenario from the simulator...")
        scenario = sim_manager.bng.get_current_scenario() # This gets the currently loaded level context
        
        print("Finding all vehicles in the scenario...")
        vehicles = scenario.find_vehicles() # Now we can find vehicles within that context
        
        player_vehicle = next((v for v in vehicles.values() if v.is_player()), None) # Find the one the user is controlling
        
        if not player_vehicle:
            raise RuntimeError("Could not find a player-controlled vehicle. Make sure you have spawned a car and can drive it.")

        print(f"Found player vehicle: {player_vehicle.vid}. Recording path now...")
        
        script = []
        player_vehicle.ai.set_mode('manual')
        
        # Loop to record position and speed nodes
        try:
            while True:
                player_vehicle.sensors.poll()
                pos = player_vehicle.state['pos']
                script.append({'x': pos[0], 'y': pos[1], 'z': pos[2], 't': 1.0})
                time.sleep(0.2)
        except KeyboardInterrupt:
            print(f"\nPath recording stopped. Recorded {len(script)} nodes.")

        # --- Part 2: Replay the path with AI and log telemetry ---
        print("\n--- Telemetry Logging Phase ---")
        print("Setting up AI replay using the same vehicle...")

        target_vehicle = player_vehicle # Reuse the vehicle we found

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
            # Poll sensors and vehicle state
            sim_manager.bng.poll_sensors_and_state()
            
            state = streamer.get_state() # This now uses the pre-polled data
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
    configs = load_configs()
    generate_target_data(configs)