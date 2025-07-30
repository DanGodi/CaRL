from beamngpy import Vehicle, sensors
import numpy as np

class TelemetryStreamer:
    """
    Attaches and polls sensors from a vehicle in BeamNG.
    This version is designed for beamngpy v1.33.1 and uses only the core 'Electrics' sensor.
    """

    def __init__(self, vehicle: Vehicle, requested_sensors: list[str]):
        """Initializes the streamer by creating and attaching the Electrics sensor."""
        self.vehicle = vehicle
        self.requested_sensors = requested_sensors
        
        # In v1.33.1, you create the sensor object first...
        self.electrics = sensors.Electrics()
        # ...and then attach it to the vehicle with a chosen name.
        self.vehicle.sensors.attach('electrics', self.electrics)
        print("TelemetryStreamer: Attached Electrics sensor.")

    def get_state(self, sensor_data) -> dict:
        """
        Processes a pre-polled Sensors object to extract required values.
        In v1.33.1, sensor_data is the vehicle.sensors object itself.
        """
        state = {}
        # --- API FIX APPLIED HERE ---
        # Access the data using dictionary-style square brackets.
        # We also add a check to make sure the sensor data exists before accessing it.
        if 'electrics' in sensor_data._sensors:
            electrics_data = sensor_data['electrics']
        else:
            electrics_data = {} # Default to empty dict if no data is present yet
        
        for key in self.requested_sensors:
            val = 0.0
            if key == 'g-force-lateral':
                val = electrics_data.get('gforce_x', 0)
            elif key == 'g-force-longitudinal':
                val = electrics_data.get('gforce_y', 0)
            elif key == 'yaw_rate':
                val = np.rad2deg(electrics_data.get('yawRate', 0))
            elif key == 'roll_rate':
                val = np.rad2deg(electrics_data.get('rollRate', 0))
            elif key == 'pitch_rate':
                val = np.rad2deg(electrics_data.get('pitchRate', 0))
            elif key == 'wheel_speed':
                val = electrics_data.get('wheelspeed', 0)
            
            state[key] = val
            
        return state

    def close(self):
        """Detaches all sensors."""
        # The detach_all method is the correct way to clean up.
        self.vehicle.sensors.detach_all()