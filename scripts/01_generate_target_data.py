import time
import pandas as pd
from pathlib import Path
import keyboard
import numpy as np  # Import numpy for distance calculation

from carlpack.beamng_control.simulation_manager import SimulationManager
from carlpack.utils.config_loader import load_configs
from carlpack.beamng_control.telemetry_streamer import TelemetryStreamer

def generate_target_data(config: dict):
    sim_manager = None
    try:
        sim_manager = SimulationManager(config['sim'])
        sim_manager.launch()

        target_model = config['sim']['target_vehicle_model']
        target_config = config['sim'].get('target_vehicle_config', None)

        sim_manager.setup_scenario(
            vehicle_model=target_model,
            vehicle_config=target_config,
            spawn_target=False
        )
        
        player_vehicle = sim_manager.base_vehicle

        print("\n" + "="*50)
        print("--- READY FOR PATH RECORDING ---")
        print("Simulator is ready. The BeamNG window is now active.")
        print(f"Drive the '{player_vehicle.model}' on the '{config['sim']['map']}' map.")
        print("\nPress 'ESC' key to stop recording.")
        print("="*50 + "\n")

        script = []
        player_vehicle.ai.set_mode('manual')
        
        sim_manager.bng.resume()
        recording_start_time = time.time()
        try:
            while not keyboard.is_pressed('esc'):
                player_vehicle.sensors.poll()
                pos = player_vehicle.state['pos']
                elapsed_time = time.time() - recording_start_time
                script.append({'x': pos[0], 'y': pos[1], 'z': pos[2], 't': elapsed_time})
                time.sleep(0.02)
        finally:
            sim_manager.bng.pause()

        if not script or len(script) < 5: # Need a few points to be a real path
            print("No valid path recorded (at least 5 points needed). Exiting.")
            return

        print(f"\nRecording stopped. Recorded {len(script)} nodes.")

        # --- Part 2: Replay the path with AI and log telemetry ---
        print("\n--- Telemetry Logging Phase ---")
        print("Setting up AI replay...")

        target_vehicle = player_vehicle

        print("Teleporting to start and setting AI script...")
        start_pos = (script[0]['x'], script[0]['y'], script[0]['z'])
        target_vehicle.teleport(start_pos, reset=True)
        
        target_vehicle.ai.set_mode('script')
        target_vehicle.ai.set_script(script)

        obs_keys = config['env']['observation_keys']
        position_keys = ['time', 'x', 'y', 'z', 'wheel_speed']
        all_keys_to_log = list(set(obs_keys + position_keys))
        
        streamer = TelemetryStreamer(target_vehicle, all_keys_to_log, bng=sim_manager.bng)
        sim_manager.bng.resume()
        
        telemetry_log = []
        start_time = time.time()
        
        # --- API FIX APPLIED HERE: The Distance Check Method ---
        # Get the final waypoint's coordinates
        last_waypoint = script[-1]
        last_pos = np.array([last_waypoint['x'], last_waypoint['y'], last_waypoint['z']])
        
        # Define how close the car needs to be to the end to be considered "finished"
        completion_threshold = 10.0  # in meters
        
        print("Logging data... This will end when the vehicle reaches the final waypoint.")
        while True:
            target_vehicle.sensors.poll()
            
            # Get the car's current position
            current_pos_dict = target_vehicle.state['pos']
            current_pos = np.array([current_pos_dict[0], current_pos_dict[1], current_pos_dict[2]])
            
            # Calculate the distance to the final waypoint
            distance_to_end = np.linalg.norm(current_pos - last_pos)
            
            # Break the loop if the car is close enough to the end
            if distance_to_end < completion_threshold:
                print("Vehicle has reached the end of the path.")
                break
                
            # Failsafe: Break if the log becomes excessively long (car is stuck)
            if len(telemetry_log) > len(script) * 10:
                print("Failsafe triggered: Log is too long. Assuming car is stuck.")
                break

            processed_state = streamer.get_state()
            
            pos = target_vehicle.state['pos']
            processed_state['time'] = time.time() - start_time
            processed_state['x'], processed_state['y'], processed_state['z'] = pos[0], pos[1], pos[2]
            
            telemetry_log.append(processed_state)
            sim_manager.bng.step(1)
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