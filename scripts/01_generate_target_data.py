# scripts/01_generate_target_data.py
import time
import pandas as pd
from pathlib import Path

from chimera.beamng_control.simulation_manager import SimulationManager
from chimera.utils.config_loader import load_configs

def generate_target_data(config: dict):
    """
    Allows the user to drive the target vehicle along a path, records it,
    and then saves the telemetry from an AI-driven replay.
    """
    sim_manager = None
    try:
        # --- Part 1: Record a path by driving manually ---
        print("--- Path Recording Phase ---")
        print("Please connect to a running instance of BeamNG.tech.")
        print("Drive the target vehicle along your desired path. The script will record it.")
        print("Press Ctrl+C in this console when you are finished driving.")

        sim_manager = SimulationManager(config['sim'])
        # We need a live connection to an already running sim for this
        sim_manager.connect() 
        
        # Find the player vehicle (assuming it's the target model)
        vehicles = sim_manager.bng.get_vehicles()
        player_vehicle = None
        for v in vehicles.values():
            if v.model == config['sim']['target_vehicle_model']:
                player_vehicle = v
                break
        
        if not player_vehicle:
            raise RuntimeError(f"Could not find a vehicle with model '{config['sim']['target_vehicle_model']}' in the simulation.")

        print(f"Found player vehicle: {player_vehicle.vid}")
        
        script = []
        player_vehicle.ai.set_mode('manual') # Ensure we have control
        
        # Loop to record position
        while True:
            try:
                player_vehicle.sensors.poll()
                pos = player_vehicle.state['pos']
                script.append({'x': pos[0], 'y': pos[1], 'z': pos[2], 't': 2.0}) # 2 sec per node
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
        
        print(f"Recorded a path with {len(script)} nodes.")
        sim_manager.close()
        time.sleep(2) # Give BeamNG time to reset

        # --- Part 2: Replay the path with AI and log telemetry ---
        print("\n--- Telemetry Logging Phase ---")
        print("Relaunching simulation to replay the recorded path with AI.")

        sim_manager = SimulationManager(config['sim'])
        sim_manager.connect()
        # Setup scenario with only the target vehicle
        sim_manager.setup_scenario(spawn_target=False)
        # Use the target vehicle we spawned, not the base
        target_vehicle = sim_manager.base_vehicle
        target_vehicle.props.model = config['sim']['target_vehicle_model']

        print("Teleporting to start and running AI script...")
        target_vehicle.teleport(tuple(script[0].values())[:3])
        sim_manager.bng.pause()
        
        # Use AI to drive the recorded path
        target_vehicle.ai.set_mode('script')
        target_vehicle.ai.set_script(script)

        # Attach sensors to log data
        from chimera.beamng_control.telemetry_streamer import TelemetryStreamer
        obs_keys = config['env']['observation_keys'] + ['time', 'x', 'y', 'z']
        streamer = TelemetryStreamer(target_vehicle, obs_keys)

        sim_manager.bng.unpause()
        telemetry_log = []
        start_time = time.time()
        
        while target_vehicle.ai.get_current_cmd_id() < len(script) - 1:
            state = streamer.get_state()
            pos = target_vehicle.state['pos']
            
            # Add position and time to the log
            state['time'] = time.time() - start_time
            state['x'], state['y'], state['z'] = pos[0], pos[1], pos[2]
            
            telemetry_log.append(state)
            time.sleep(1 / 30) # Log at ~30Hz

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