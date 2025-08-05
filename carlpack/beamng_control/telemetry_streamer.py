# carlpack/beamng_control/telemetry_streamer.py
# --- MODIFIED VERSION with AdvancedIMU ---

from beamngpy import Vehicle, sensors
import numpy as np

class TelemetryStreamer:
    """
    Attaches and polls sensors from a vehicle in BeamNG.
    This version uses the AdvancedIMU for rotational/G-force data and Electrics for other data.
    """

    def __init__(self, vehicle: Vehicle, requested_sensors: list[str], bng: 'BeamNGpy'):
        """Initializes the streamer by creating and attaching all necessary sensors."""
        self.vehicle = vehicle
        self.requested_sensors = requested_sensors
        self.bng = bng # Store the bng instance, required by AdvancedIMU

        print("TelemetryStreamer: Attaching required sensors...")

        # Keep the Electrics sensor, it's simple and reliable
        self.electrics = sensors.Electrics()
        self.vehicle.sensors.attach('electrics', self.electrics)
        
      # --- MODIFICATION START ---
        # Add the AdvancedIMU sensor with a specific update time. This is critical for
        # ensuring the sensor generates data in a synchronous environment.
        self.advanced_imu = sensors.AdvancedIMU(
            'advanced_imu',
            self.bng,
            self.vehicle,
            gfx_update_time=0.01  # Request updates at 100Hz
        )

        print("TelemetryStreamer: All sensors created and attached.")

    def get_state(self, sensor_data) -> dict:
        """
        Processes a pre-polled Sensors object to extract required values.
        """
        state = {}
        
        # Safely get data from both sensor dictionaries
        electrics_data = sensor_data['electrics'] if 'electrics' in sensor_data._sensors else {}
        imu_data = sensor_data['advanced_imu'] if 'advanced_imu' in sensor_data._sensors else {}
        
        for key in self.requested_sensors:
            val = 0.0
            
            # --- MODIFICATION START: Re-routing data sources ---
            if key == 'g-force-lateral':
                # Data now comes from AdvancedIMU's 'accSmooth' (Y-axis)
                val = imu_data.get('accSmooth', [0, 0, 0])[1]
            elif key == 'g-force-longitudinal':
                # Data now comes from AdvancedIMU's 'accSmooth' (X-axis)
                val = imu_data.get('accSmooth', [0, 0, 0])[0]
            elif key == 'yaw_rate':
                # Data now comes from AdvancedIMU's 'angVelSmooth' (Z-axis)
                val = np.rad2deg(imu_data.get('angVelSmooth', [0, 0, 0])[2])
            elif key == 'roll_rate':
                # Data now comes from AdvancedIMU's 'angVelSmooth' (X-axis)
                val = np.rad2deg(imu_data.get('angVelSmooth', [0, 0, 0])[0])
            elif key == 'pitch_rate':
                # Data now comes from AdvancedIMU's 'angVelSmooth' (Y-axis)
                val = np.rad2deg(imu_data.get('angVelSmooth', [0, 0, 0])[1])
            
            # As requested, 'wheel_speed' has been removed.
            # --- MODIFICATION END ---
            
            state[key] = val
            
        return state

    def close(self):
        """Detaches all sensors."""
        # This will correctly clean up both the electrics and imu sensors
        self.vehicle.sensors.detach_all()