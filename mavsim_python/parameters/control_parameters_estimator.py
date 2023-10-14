import sys
sys.path.append('..')
import numpy as np
import design_projects.chap05.model_coef as TF
import parameters.aerosonde_parameters as MAV


#### TODO #####
gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim  # Vg0 = Va0 + Vw0, Vw0 is steady-state wind  Assume zero steady-state wind
Vg0 = Va0
rho = MAV.rho # density of air XXX shouldn't this be a function of altitude, like in STD76?
sigma = 0  # low pass filter gain for derivative

#----------roll loop-------------
# get transfer function data for delta_a to phi
wn_roll = 8
zeta_roll = 0.8
roll_kp = wn_roll**2 / TF.a_phi2
roll_kd = (2 * zeta_roll * wn_roll - TF.a_phi1) / TF.a_phi2

#----------course loop-------------
W_course = 15  # XXX >= 1, book recommends between 5 and 10
wn_course = wn_roll / W_course
zeta_course = 1.5
course_kp = 2 * zeta_course * wn_course * Vg0 / gravity
course_ki = wn_course**2 * Vg0 / gravity

#----------yaw damper-------------
yaw_damper_p_wo = 0.5
yaw_damper_kr = 0.5

#----------pitch loop-------------
wn_pitch = 40  # XXX if this number is too small, the plane will fall out of the sky (literally)
zeta_pitch = 1.3
pitch_kp = (wn_pitch**2 - TF.a_theta2) / TF.a_theta3
pitch_kd = (2. * zeta_pitch * wn_pitch - TF.a_theta1) / TF.a_theta3
K_theta_DC = pitch_kp * TF.a_theta3 / (TF.a_theta2 + pitch_kp * TF.a_theta3)

#----------altitude loop-------------
W_altitude = 50  # XXX >= 1, book recommends between 5 and 10, too small and the step response rings a lot
wn_altitude = wn_pitch / W_altitude
zeta_altitude = 1.
altitude_kp = 2 * zeta_altitude * wn_altitude / (K_theta_DC * Va0) 
altitude_ki = wn_altitude**2 / (K_theta_DC * Va0)

#---------airspeed hold using throttle---------------
wn_airspeed_throttle = 5.
zeta_airspeed_throttle = np.sqrt(2)
airspeed_throttle_kp = (2 * zeta_airspeed_throttle * wn_airspeed_throttle - TF.a_V1) / TF.a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2 / TF.a_V2

#---------controller limits--------------------
max_aileron = np.radians(45)
max_rudder = np.radians(30)
max_elevator = np.radians(45)

#---------controller limits--------------------
max_roll = np.radians(45)
max_pitch = np.radians(30)
altitude_zone = 15  # h_hold from book/notes

## LQR
max_delta_sideslip = np.radians(10)
max_delta_p = 5
max_delta_r = max_delta_p
max_delta_phi = np.radians(30)
max_delta_chi = np.radians(90)
max_settle_time_chi = 5
max_settle_time_sideslip = 5  # if this value is too large, the control response will be sluggish 
max_sideslip_int = 0.5 * max_delta_sideslip * max_settle_time_sideslip  # area of a triangle with base == settle time and height == max angle (expected worst-case step delta)
max_chi_int = 0.5 * max_delta_chi * max_settle_time_chi
max_delta_airspeed = 20
max_delta_alpha = np.radians(20)
max_delta_q = 5
max_delta_theta = np.radians(30)
max_delta_altitude = 50
max_settle_time_altitude = 5
max_settle_time_airspeed = 5
max_altitude_int = 0.5 * max_delta_altitude * max_settle_time_altitude
max_airspeed_int = 0.5 * max_delta_airspeed * max_settle_time_airspeed

## TECS
k_T_tecs = 0.2
k_D_tecs = 0.45  # recommended > k_T 
k_Va_tecs = 0.5  # 1 / time constant for first-order acceleration model
k_h_tecs = 0.5  # 1 / time constant for first-order climb rate model
wn_pitch_tecs = 20
zeta_pitch_tecs = 1.0
pitch_kp_tecs = (wn_pitch_tecs**2 - TF.a_theta2) / TF.a_theta3
pitch_kd_tecs = (2. * zeta_pitch_tecs * wn_pitch_tecs - TF.a_theta1) / TF.a_theta3
## PI corrections for modeling errors
throttle_correction_kp_tecs = 0.05
throttle_correction_ki_tecs = 0.05
throttle_correction_limit = 0.1
altitude_correction_kp_tecs = 0.05
altitude_correction_ki_tecs = 0.05
altitude_correction_limit = np.radians(5)

