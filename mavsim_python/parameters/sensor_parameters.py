import sys
sys.path.append('..')
import numpy as np
from tools.rotations import Euler2Rotation

np.random.seed(11011)

#-------- Accelerometer --------
accel_sigma = 0.0025*9.81  # standard deviation of accelerometers in m/s^2

#-------- Rate Gyro --------
gyro_bias = np.radians(5.) 
gyro_x_bias = gyro_bias * np.random.uniform(-1, 1)  # bias on x_gyro
gyro_y_bias = gyro_bias * np.random.uniform(-1, 1)  # bias on y_gyro
gyro_z_bias = gyro_bias * np.random.uniform(-1, 1)  # bias on z_gyro
gyro_sigma = np.radians(0.13)  # standard deviation of gyros in rad/sec

#-------- Pressure Sensor(Altitude) --------
abs_pres_bias = 0.125 * 1000 * np.random.uniform(-1, 1)  # from Appendix H
abs_pres_sigma = 0.01 * 1000  # standard deviation of absolute pressure sensors in Pascals

#-------- Pressure Sensor (Airspeed) --------
diff_pres_bias = 0.020*1000  * np.random.uniform(-1, 1) # from Appendix H
diff_pres_sigma = 0.002*1000  # standard deviation of diff pressure sensor in Pascals

#-------- Magnetometer --------
# take inclination and declination for Provo, UT from the accompanying slides for Chapter 7: https://drive.google.com/file/d/1BMceIPDGzBda9w5R5LrNnouabS8lQ9s_/view (slide 20)
mag_inclination_deg = 66
mag_declination_deg = 12.5
# magnetic to inertial transformation
R_m2i = Euler2Rotation(0., -np.radians(mag_inclination_deg), np.radians(mag_declination_deg)) 
mag_bias = 0.006  # results in ~1 deg heading bias, as specified in the book
mag_x_bias = mag_bias * np.random.uniform(-1, 1)  # bias on x_mag
mag_y_bias = mag_bias * np.random.uniform(-1, 1)  # bias on y_mag
mag_z_bias = mag_bias * np.random.uniform(-1, 1)  # bias on z_mag
mag_sigma = 0.0045 / 3  # results in ~0.3 deg 1 sigma for heading, as specified in the book
mag_heading_sigma = np.radians(0.3)  # inflate the error while the bias estimates settle

# #-------- GPS --------
# ts_gps = 1.0
# gps_k = 1. / 1100.  # 1 / s
# gps_n_sigma = 0.21
# gps_e_sigma = 0.21
# gps_h_sigma = 0.40
# gps_Vg_sigma = 0.05
# gps_course_sigma = gps_Vg_sigma / 10

#-------- 2017 GPS --------
ts_gps = 0.2
gps_k = 1. / 1100.  # 1 / s
gps_n_sigma = 0.01
gps_e_sigma = 0.01
gps_h_sigma = 0.03
gps_Vg_sigma = 0.005
#gps_course_sigma = gps_Vg_sigma / 20  # this depends on the ground speed
#gps_chi_sigma = gps_Vg_sigma / 20  # this depends on the ground speed
