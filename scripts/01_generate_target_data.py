# scripts/01_generate_target_data.py
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
        # --- Part 1: Record a path by driving manually ---
        print("--- Path Recording Phase ---")
        print("Please connect to a running instance of BeamNG.tech.")
        print("Drive the target vehicle along your desired path.")
        print("Press Ctrl+C in this console when you are finished driving.")

        sim_manager = SimulationManager(config['sim'])
        sim_manager.connect()
        
        vehicles = sim_manager.bng.get_vehicles()
        player_vehicle = next((v for v in vehicles.values() if v.is_player()), None)
        
        if not player_vehicle:
            raise RuntimeError("Could not find a player-controlled vehicle in the simulation.")

        print(f"Found player vehicle: {player_vehicle.vid}. Recording path now...")
        
        script = []
        player_vehicle.ai.set_mode('manual')
        
        # Loop to record position and speed nodes
        try:
            while True:
                player_vehicle.sensors.poll()
                pos = player_vehicle.state['pos']
                # Record position and a default speed for the AI to target
                # A more advanced version could record the actual speed here too.
                script.append({'x': pos[0], 'y': pos[1], 'z': pos[2], 't': 1.0})
                time.sleep(0.2) # Record a node every 200ms
        except KeyboardInterrupt:
            print(f"\nPath recording stopped. Recorded {len(script)} nodes.")

        sim_manager.close()
        time.sleep(2) # Give BeamNG time to reset if needed

        # --- Part 2: Replay the path with AI and log telemetry ---
        print("\n--- Telemetry Logging Phase ---")
        print("Relaunching simulation to replay the recorded path with AI.")

        sim_manager = SimulationManager(config['sim'])
        sim_manager.connect()
        sim_manager.setup_scenario(spawn_target=False) # Spawn a fresh vehicle
        
        # We use the spawned vehicle as our target vehicle for the replay
        target_vehicle = sim_manager.base_vehicle 
        # Ensure it's the correct model from config
        target_vehicle.props.model = config['sim']['target_vehicle_model']

        print("Teleporting to start and setting AI script...")
        start_pos = (script[0]['x'], script[0]['y'], script[0]['z'])
        target_vehicle.teleport(start_pos)
        
        target_vehicle.ai.set_mode('script', looping=False)
        target_vehicle.ai.set_script(script)

        # Attach sensors to log data
        obs_keys = config['env']['observation_keys']
        # We also want to log speed and position to use in the environment
        position_keys = ['time', 'x', 'y', 'z', 'wheel_speed']
        all_keys_to_log = list(set(obs_keys + position_keys))
        
        streamer = TelemetryStreamer(target_vehicle, all_keys_to_log)

        sim_manager.bng.unpause()
        telemetry_log = []
        start_time = time.time()
        
        print("Logging data...")
        # Log until the AI has completed the script
        while target_vehicle.ai.get_current_cmd_id() < len(script) - 1:
            state = streamer.get_state()
            pos = target_vehicle.state['pos']
            
            state['time'] = time.time() - start_time
            state['x'], state['y'], state['z'] = pos[0], pos[1], pos[2]
            
            telemetry_log.append(state)
            time.sleep(1 / 50) # Log at ~50Hz

        print(f"Logged {len(telemetry_log)} telemetry points.")
        
        # Save to CSV
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