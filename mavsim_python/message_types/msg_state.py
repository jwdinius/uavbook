"""
msgState 
    - messages type for state, that will be passed between blocks in the architecture
    
part of mavPySim 
    - Beard & McLain, PUP, 2012
    - Update history:  
        1/9/2019 - RWB
        3/30/2022 - RWB
"""
import numpy as np

MAX_WIND = 5.

class MsgState:
    def __init__(self):
        self.north = 0.      # inertial north position in meters
        self.east = 0.      # inertial east position in meters
        self.altitude = 100.       # inertial altitude in meters
        self.phi = 0.     # roll angle in radians
        self.theta = 0.   # pitch angle in radians
        self.psi = 0.     # yaw angle in radians
        self.Va = 25.      # airspeed in meters/sec
        self.alpha = 0.   # angle of attack in radians
        self.beta = 0.    # sideslip angle in radians
        self.p = 0.       # roll rate in radians/sec
        self.q = 0.       # pitch rate in radians/sec
        self.r = 0.       # yaw rate in radians/sec
        self.Vg = 25.      # groundspeed in meters/sec
        self.gamma = 0.   # flight path angle in radians
        self.chi = 0.     # course angle in radians
        self.wn = 0.      # inertial windspeed in north direction in meters/sec
        self.we = 0.      # inertial windspeed in east direction in meters/sec
        self.wd = 0.      # inertial windspeed in east direction in meters/sec
        self.bx = 0.      # gyro bias along roll axis in radians/sec
        self.by = 0.      # gyro bias along pitch axis in radians/sec
        self.bz = 0.      # gyro bias along yaw axis in radians/sec
        self.camera_az = 0.  # camera azimuth angle
        self.camera_el = np.radians(-90)  # camera elevation angle
        
        self.sigma_north = 0.5      # inertial north position uncertainty (1-sigma) in meters
        self.sigma_east = 0.5      # inertial east position uncertainty (1-sigma) in meters
        self.sigma_altitude = 0.5       # inertial altitude uncertainty (1-sigma) in meters
        self.sigma_phi = np.radians(5)     # roll angle uncertainty (1-sigma) in radians
        self.sigma_theta = np.radians(5)   # pitch angle uncertainty (1-sigma) in radians
        self.sigma_psi = np.radians(5)     # yaw angle uncertainty in radians
        self.sigma_Va = 1.      # airspeed uncertainty in meters/sec
        self.sigma_alpha = np.radians(5)   # angle of attack uncertainty (1-sigma) in radians
        self.sigma_beta = np.radians(5)    # sideslip angle uncertainty (1-sigma) in radians
        self.sigma_p = np.radians(5)       # roll rate uncertainty (1-sigma) in radians/sec
        self.sigma_q = np.radians(5)       # pitch rate uncertainty (1-sigma) in radians/sec
        self.sigma_r = np.radians(5)       # yaw rate uncertainty (1-sigma) in radians/sec
        self.sigma_Vg = 1.      # groundspeed uncertainty (1-sigma) in meters/sec
        self.sigma_gamma = np.radians(5)   # flight path uncertainty (1-sigma) angle in radians
        self.sigma_chi = np.radians(5)     # course angle uncertainty (1-sigma) in radians
        self.sigma_wn = MAX_WIND / np.sqrt(3)      # inertial windspeed uncertainty in north (1-sigma) direction in meters/sec
        self.sigma_we = MAX_WIND / np.sqrt(3)      # inertial windspeed uncertainty in east (1-sigma) direction in meters/sec
        self.sigma_wd = MAX_WIND / np.sqrt(3)      # inertial windspeed uncertainty in down (1-sigma) direction in meters/sec
        self.sigma_bx = 0.      # gyro bias uncertainty along roll axis (1-sigma) in radians/sec
        self.sigma_by = 0.      # gyro bias uncertainty along pitch axis (1-sigma) in radians/sec
        self.sigma_bz = 0.      # gyro bias uncertainty along yaw axis (1-sigma) in radians/sec
