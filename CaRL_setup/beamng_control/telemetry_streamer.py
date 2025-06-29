# chimera/beamng_control/telemetry_streamer.py
from beamngpy import Vehicle, sensors
import numpy as np

class TelemetryStreamer:
    """Attaches and polls sensors from a vehicle in BeamNG."""

    def __init__(self, vehicle: Vehicle, requested_sensors: list[str]):
        """
        Initializes and attaches the necessary sensors to the vehicle.

        Args:
            vehicle: The beamngpy.Vehicle object to monitor.
            requested_sensors: A list of sensor names to collect.
        """
        self.vehicle = vehicle
        self.requested_sensors = requested_sensors
        self.sensors = {}

        # The Electrics sensor is the primary way to get most telemetry
        electrics = sensors.Electrics()
        self.sensors['electrics'] = electrics
        
        # Some values like G-forces require a specific sensor (IMU)
        if any('g-force' in s for s in requested_sensors):
            imu = sensors.IMU(
                pos=(0, 0, 1.7), # Position relative to vehicle's origin
                # I AM NOT SURE IF THIS ORIENTATION IS CORRECT.
                # IT MIGHT NEED ROTATION TO ALIGN WITH THE VEHICLE AXES.
                # USE (0, 0, 0) AND (1, 0, 0, 0) AND DEBUG THE OUTPUT.
                dir=(0, 0, -1), up=(0, -1, 0)
            )
            self.sensors['imu'] = imu
        
        # Attach all created sensors to the vehicle
        self.vehicle.sensors.attach_many(self.sensors)

    def get_state(self) -> dict:
        """
        Polls the attached sensors and returns the latest vehicle state.
        
        Returns:
            A dictionary where keys are sensor names and values are the readings.
        """
        # Poll the vehicle to get the latest sensor data
        sensor_data = self.vehicle.sensors.poll()
        
        state = {}
        # Extract the required values from the polled data
        for key in self.requested_sensors:
            val = 0.0 # Default value if sensor data is missing
            if key == 'g-force-lateral' and 'imu' in sensor_data:
                val = sensor_data['imu']['accX']
            elif key == 'g-force-longitudinal' and 'imu' in sensor_data:
                val = sensor_data['imu']['accY']
            elif key == 'yaw_rate' and 'electrics' in sensor_data:
                # Convert from rad/s to deg/s
                val = np.rad2deg(sensor_data['electrics'].get('yawRate', 0))
            elif key == 'roll_rate' and 'electrics' in sensor_data:
                val = np.rad2deg(sensor_data['electrics'].get('rollRate', 0))
            elif key == 'pitch_rate' and 'electrics' in sensor_data:
                val = np.rad2deg(sensor_data['electrics'].get('pitchRate', 0))
            elif key == 'wheel_speed' and 'electrics' in sensor_data:
                # Average speed of all wheels
                val = sensor_data['electrics'].get('wheelspeed', 0)
            # Add other custom sensors here
            
            state[key] = val
            
        return state

    def close(self):
        """Detaches all sensors."""
        self.vehicle.sensors.detach_all()