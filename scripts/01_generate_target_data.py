import time
import pandas as pd
from pathlib import Path
import keyboard
import numpy as np 
from beamngpy import sensors

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

        obs_keys = config['env']['observation_keys']
        position_keys = ['time', 'x', 'y', 'z', 'wheel_speed']
        all_keys_to_log = list(set(obs_keys + position_keys))
        
        player_vehicle = sim_manager.base_vehicle
        streamer = TelemetryStreamer(player_vehicle, all_keys_to_log, bng=sim_manager.bng)

        print("\n" + "="*50)
        print("--- READY FOR PATH RECORDING ---")
        print("Simulator is ready. The BeamNG window is now active.")
        print(f"Drive the '{player_vehicle.model}' on the '{config['sim']['map']}' map.")
        print("\nPress 'ESC' key to stop recording.")
        print("="*50 + "\n")

        script = []
        player_vehicle.ai.set_mode('manual')
        
        sim_manager.bng.resume()
        try:
            while not keyboard.is_pressed('esc'):
                player_vehicle.sensors.poll()
                pos = player_vehicle.state['pos']
                state = streamer.get_state()
                sim_time = state['time']
                script.append({'x': pos[0], 'y': pos[1], 'z': pos[2], 't': sim_time})
                time.sleep(1/50)

        finally:
            sim_manager.bng.pause()

        if not script or len(script) < 5:
            print("No valid path recorded (at least 5 points needed). Exiting.")
            return

        print(f"\nRecording stopped. Recorded {len(script)} nodes.")

        print("\n--- Telemetry Logging Phase ---")
        print("Setting up AI replay...")

        target_vehicle = player_vehicle

        print("Teleporting to start and setting AI script...")
        start_pos = (script[0]['x'], script[0]['y'], script[0]['z'])
        target_vehicle.teleport(start_pos, rot_quat=(0, 0, 1, 0), reset=True)
        target_vehicle.ai.set_mode('script')
        target_vehicle.ai.set_script(script)

        sim_manager.bng.resume()
        
        telemetry_log = []
        
        last_waypoint = script[-1]
        last_pos = np.array([last_waypoint['x'], last_waypoint['y'], last_waypoint['z']])
        
        completion_threshold = 5.0  # in meters
        
        print("Logging data... This will end when the vehicle reaches the final waypoint.")
        reset_time = state['time']
        while True:
            target_vehicle.sensors.poll()
            
            current_pos_dict = target_vehicle.state['pos']
            current_pos = np.array([current_pos_dict[0], current_pos_dict[1], current_pos_dict[2]])
            
            distance_to_end = np.linalg.norm(current_pos - last_pos)
            
            if distance_to_end < completion_threshold:
                print("Vehicle has reached the end of the path.")
                break
                
            if len(telemetry_log) > len(script) * 10:
                print("Failsafe triggered: Log is too long. Assuming car is stuck.")
                break

            processed_state = streamer.get_state()
            processed_state['time'] = processed_state['time'] - reset_time

            pos = target_vehicle.state['pos']
            processed_state['x'], processed_state['y'], processed_state['z'] = pos[0], pos[1], pos[2]
            
            telemetry_log.append(processed_state)
            time.sleep(1/50)

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