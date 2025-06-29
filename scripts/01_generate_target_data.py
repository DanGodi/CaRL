# scripts/01_generate_target_data.py
import time
import pandas as pd
from pathlib import Path

from chimera.beamng_control.simulation_manager import SimulationManager
from chimera.utils.config_loader import load_configs
from chimera.beamng_control.telemetry_streamer import TelemetryStreamer

def generate_target_data(config: dict):
    """
    Allows the user to drive the target vehicle, recording both the resulting telemetry
    AND the driver's inputs (steering, throttle, brake).
    """
    sim_manager = None
    try:
        print("--- Target Data Recording ---")
        print("Please connect to a running instance of BeamNG.tech.")
        print("Drive the target vehicle along your desired path.")
        print("Press Ctrl+C in this console when you are finished driving.")

        sim_manager = SimulationManager(config['sim'])
        sim_manager.connect()
        
        # Find the player vehicle
        vehicles = sim_manager.bng.get_vehicles()
        player_vehicle = None
        for v in vehicles.values():
            # Find the vehicle currently being controlled by the player
            if v.is_player():
                player_vehicle = v
                break
        
        if not player_vehicle:
            raise RuntimeError("Could not find a player-controlled vehicle in the simulation.")

        print(f"Found player vehicle: {player_vehicle.vid}. Model: {player_vehicle.model}")
        
        # We need to record the driver's inputs as well.
        # Add these to the list of things to log.
        obs_keys = config['env']['observation_keys']
        input_keys = ['steering_input', 'throttle_input', 'brake_input']
        position_keys = ['time', 'x', 'y', 'z']
        all_keys_to_log = obs_keys + input_keys + position_keys
        
        streamer = TelemetryStreamer(player_vehicle, all_keys_to_log)
        
        telemetry_log = []
        start_time = time.time()
        print("Starting log... Drive now.")

        while True:
            try:
                # Get all sensor data
                state = streamer.get_state()
                pos = player_vehicle.state['pos']
                
                # Add position and time to the log
                state['time'] = time.time() - start_time
                state['x'], state['y'], state['z'] = pos[0], pos[1], pos[2]
                
                telemetry_log.append(state)
                # Log at a reasonably high frequency
                time.sleep(1 / 50) 
            except KeyboardInterrupt:
                break
        
        print(f"\nLogged {len(telemetry_log)} telemetry points.")
        
        # Save to CSV
        output_path = Path(config['sim']['target_data_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(telemetry_log)
        df.to_csv(output_path, index=False)
        print(f"Target telemetry and input data saved to: {output_path}")

    finally:
        if sim_manager:
            sim_manager.close()

if __name__ == '__main__':
    configs = load_configs()
    generate_target_data(configs)