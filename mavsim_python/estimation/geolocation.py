"""
target geolocation algorithm
    - Beard & McLain, PUP, 2012
    - Updated:
        4/1/2022 - RWB
        4/6/2022 - RWB
"""
import numpy as np
from scipy import stats
import parameters.simulation_parameters as SIM
import parameters.camera_parameters as CAM
import parameters.planning_parameters as PLAN
from tools.rotations import Euler2Rotation

ACCEL_PROCESS_NOISE = 2.0
POS_MEAS_NOISE = 0.5

# XXX BROKEN AT THE MOMENT

class Geolocation:
    def __init__(self, xhat, ts_control):
        # initialize EKF for geolocation

        ###### TODO ######
        self.xhat = xhat.reshape((4, 1))
        self.Q = np.diag([
            (ACCEL_PROCESS_NOISE*ts_control)**2,
            (ACCEL_PROCESS_NOISE*ts_control)**2,
            (ACCEL_PROCESS_NOISE*ts_control)**2,
            (ACCEL_PROCESS_NOISE)**2,
            (ACCEL_PROCESS_NOISE)**2,
            (ACCEL_PROCESS_NOISE)**2,
            3*(ACCEL_PROCESS_NOISE*ts_control)**2
        ])
        self.R = np.diag([
            POS_MEAS_NOISE**2,
            POS_MEAS_NOISE**2,
            POS_MEAS_NOISE**2,
            3*POS_MEAS_NOISE**2
        ])
        self.N = 1  # number of prediction step per sample
        self.P = np.diag([
            (PLAN.city_width/3)**2,
            (PLAN.city_width/3)**2,
            (PLAN.city_width/3)**2,
            PLAN.city_width**2 / 3
        ])
        self.gate_threshold = stats.chi2.isf(q=0.05, df=4)
        self.Ts = SIM.ts_control/self.N
        self.Q_scaling = 1
        # initialize viewer for geolocation error

    def update(self, mav_state, pixels):
        self.propagate_model(mav_state)
        self.measurement_update(mav_state, pixels)
        return self.xhat[0:3, :]  # return estimated NED position

    def propagate_model(self, mav_state):
        # model propagation
        Tp = self.Ts / self.N
        for _ in range(0, self.N):
            A = jacobian(self.f, self.xhat, mav_state)
            self.xhat += self.f(self.xhat, mav_state) * Tp
            
            # account for error in north, east, altitude, Vg, chi, and gamma here
            x_s = np.array([
                [mav_state.north],
                [mav_state.east],
                [-mav_state.altitude],
                [mav_state.Vg],
                [mav_state.chi],
                [mav_state.gamma]
            ])
            J_s = jacobian(self.f_s, x_s, self.xhat)
            Q_s = J_s @ np.diag([
                mav_state.sigma_north**2,
                mav_state.sigma_east**2,
                mav_state.sigma_altitude**2,
                mav_state.sigma_Vg**2,
                mav_state.sigma_chi**2,
                mav_state.sigma_gamma**2
            ]) @ J_s.T

            # total process noise = model error + "state" members  errors
            Q = self.Q_scaling * (self.Q + Q_s)
            self.P += (A @ self.P + self.P @ A.T + Q) * Tp


    def measurement_update(self, mav_state, pixels):
        # measurement updates
        h = self.h(self.xhat, mav_state)
        C = jacobian(self.h, self.xhat, mav_state)
        y = self.measurements(mav_state, pixels)
        ###### TODO ######
        # self.P = ?
        # self.xhat = ?
        pass

    def f(self, xhat, mav):
        target_position = xhat[:3, 0]
        target_velocity = xhat[3:6, 0]
        
        ######  TODO  ######
        p_mav_i = np.array([[mav.north], [mav.east], [-mav.altitude]])
        v_mav_i = np.array([
            [mav.Vg * np.cos(mav.chi) * np.cos(mav.gamma)], 
            [mav.Vg * np.sin(mav.chi) * np.cos(mav.gamma)], 
            [-mav.Vg * np.sin(mav.gamma)]
        ])
        # system dynamics for propagation model: xdot = f(x, u)
        f_ = np.zeros((7, 1))
        # position dot = velocity
        v = target_position - p_mav_i
        f_[:3, 0] = v
        
        # velocity dot = zero
        f_[3:6, 0] = target_velocity - v_mav_i 

        # Ldot
        f_[6, 0] = -float( (v.T @ v_mav_i) / xhat[6, 0] ) 

        return f_
    
    def f_s(self, xhat, mav):
        # invert: x<->state
        target_position = mav[:3, 0]
        target_velocity = mav[3:6, 0]

        n, e, d = [xhat.item(i) for i in range(0, 3)]
        Vg, chi, gamma = [xhat.item(i) for i in range(3, 6)]

        ######  TODO  ######
        p_mav_i = np.array([[n], [e], [d]])
        v_mav_i = np.array([
            [Vg * np.cos(chi) * np.cos(gamma)], 
            [Vg * np.sin(chi) * np.cos(gamma)], 
            [-Vg * np.sin(gamma)]
        ])
        # system dynamics for propagation model: xdot = f(x, u)
        f_ = np.zeros((7, 1))
        # position dot = velocity
        v = target_position - p_mav_i
        f_[:3, 0] = v
        
        # velocity dot = zero
        f_[3:6, 0] = target_velocity - v_mav_i 

        # Ldot
        f_[6, 0] = -float( (v.T @ v_mav_i) / mav[6, 0] ) 

        return f_

    def h(self, xhat, mav):
        ###### TODO ######
        target_position = xhat[:3, 0]
        p_mav_i = np.array([[mav.north], [mav.east], [-mav.altitude]])
        L = xhat[6, 0]

        l_i = target_position - p_mav_i
        l_i /= np.linalg.norm(l_i)

        return target_position

    def measurements(self, mav, pixels):
        ####### TODO ########
        p_mav_i = np.array([[mav.north], [mav.east], [-mav.altitude]])
        
        ex, ey = [pixels.item(i) for i in range(2)]
        l_c = np.array([[ex], [ey], [CAM.f]])
        F = np.linalg.norm(l_c)
        l_c /= F 

        R_g2b = Euler2Rotation(0, mav.gimbal_el, mav.gimbal_az) 
        R_c2g = np.array(
            [[0], [0], [1]],
            [[1], [0], [0]],
            [[0], [1], [0]]
        )
        R_c2b = R_g2b @ R_c2g

        l_b = R_c2b @ l_c
        R_b2i = Euler2Rotation(mav.phi, mav.theta, mav.psi)

        l_i = R_b2i @ l_b

        # "measure" L using eq. 13.17
        L = mav.altitude / l_i.item(2)

        return p_mav_i + L * l_i

def jacobian(fun, x, mav_state):
    # compute jacobian of fun with respect to x
    f = fun(x, mav_state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, mav_state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J
