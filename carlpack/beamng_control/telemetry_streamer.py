from beamngpy import Vehicle, sensors
import numpy as np

class TelemetryStreamer:
    """
    Attaches and polls the AdvancedIMU sensor to get all core vehicle dynamics data.
    """

    def __init__(self, vehicle: Vehicle, requested_sensors: list[str], bng: 'BeamNGpy'):
        """Initializes the streamer by creating and attaching the AdvancedIMU sensor."""
        self.vehicle = vehicle
        self.requested_sensors = requested_sensors
        self.bng = bng
        
        print("TelemetryStreamer: Attaching AdvancedIMU sensor...")
        
        # Create the AdvancedIMU sensor. Its constructor handles attachment.
        self.advanced_imu = sensors.AdvancedIMU('advanced_imu', self.bng, self.vehicle)

        print("TelemetryStreamer: Sensor attached.")

    def get_state(self) -> dict:
        """
        Processes sensor data by explicitly polling the AdvancedIMU and
        correctly parsing its dictionary-of-dictionaries output format.
        """
        state = {}
        
        # Explicitly poll the AdvancedIMU sensor to get its data.
        imu_poll_result = self.advanced_imu.poll()
        
        # --- THE DEFINITIVE FIX ---
        # The poll result is a dictionary where keys are stringified numbers ("0.0", "1.0", etc.)
        # and values are the reading dictionaries. We need the MOST RECENT reading.
        if imu_poll_result:
            # Find the key for the last reading (e.g., "5.0") by converting keys to float for comparison.
            last_reading_key = max(imu_poll_result.keys(), key=float)
            # Get the dictionary for that last reading.
            imu_data = imu_poll_result[last_reading_key]
        else:
            # Default to an empty dictionary if the poll returns nothing.
            imu_data = {}
        # --- END OF FIX ---
        
        for key in self.requested_sensors:
            val = 0.0
            if key == 'g-force-lateral':
                # Lateral (side-to-side) Gs come from the IMU's Y-axis of accSmooth.
                val = imu_data.get('accSmooth', [0, 0, 0])[1]
            elif key == 'g-force-longitudinal':
                # Longitudinal (front-back) Gs come from the IMU's X-axis of accSmooth.
                val = imu_data.get('accSmooth', [0, 0, 0])[0]
            elif key == 'yaw_rate':
                # Yaw rate comes from the IMU's Z-axis angular velocity.
                val = np.rad2deg(imu_data.get('angVelSmooth', [0, 0, 0])[2])
            elif key == 'roll_rate':
                # Roll rate comes from the IMU's X-axis angular velocity.
                val = np.rad2deg(imu_data.get('angVelSmooth', [0, 0, 0])[0])
            elif key == 'pitch_rate':
                # Pitch rate comes from the IMU's Y-axis angular velocity.
                val = np.rad2deg(imu_data.get('angVelSmooth', [0, 0, 0])[1])
            elif key == 'time':
                val = imu_data.get('time', 0.0)
            elif key == 'y':
                val = imu_data.get('pos', [0, 0, 0])[1]
            elif key == 'x':
                val = imu_data.get('pos', [0, 0, 0])[0]

            state[key] = val
            
        return state

    def close(self):
        """Sensors are managed by the vehicle object."""
        pass
