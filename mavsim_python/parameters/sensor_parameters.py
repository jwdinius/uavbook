import sys
sys.path.append('..')
import numpy as np

np.random.seed(11011)

#-------- Accelerometer --------
accel_sigma = 0.0025*9.81  # standard deviation of accelerometers in m/s^2

#-------- Rate Gyro --------
gyro_x_bias = np.radians(5*np.random.uniform(-1, 1))  # bias on x_gyro
gyro_y_bias = np.radians(5*np.random.uniform(-1, 1))  # bias on y_gyro
gyro_z_bias = np.radians(5*np.random.uniform(-1, 1))  # bias on z_gyro
gyro_sigma = np.radians(0.13)  # standard deviation of gyros in rad/sec

#-------- Pressure Sensor(Altitude) --------
abs_pres_bias = 0.125*1000  # from Appendix H
abs_pres_sigma = 0.01*1000  # standard deviation of absolute pressure sensors in Pascals

#-------- Pressure Sensor (Airspeed) --------
diff_pres_bias = 0.020*1000  # from Appendix H
diff_pres_sigma = 0.002*1000  # standard deviation of diff pressure sensor in Pascals

#-------- Magnetometer --------
neg_inclination_angle = -np.radians(65.7)
cni, sni = np.cos(neg_inclination_angle), np.sin(neg_inclination_angle)
declination_angle = np.radians(12.12)  # take inclination and declination for Provo, UT from book
cd, sd = np.cos(declination_angle), np.sin(declination_angle)
R_mag_to_inertial = np.array([[cni, 0, -sni], [0, 1, 0], [sni, 0, cni]]) @ np.array([[cd, sd, 0], [-sd, cd, 0], [0, 0, 1]])
mag_bias = np.radians(1.0) / np.sqrt(3)
mag_sigma = np.radians(0.03) / np.sqrt(3)  # assume bias and sigma are distributed equally along body axes

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
gps_course_sigma = gps_Vg_sigma / 20
