"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
from scipy import stats
sys.path.append('..')
import parameters.aerosonde_parameters as MAV
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from tools.rotations import Euler2Rotation
from message_types.msg_state import MsgState, MAX_WIND
from message_types.msg_sensors import MsgSensors
from scipy.optimize import minimize

SIGMA_FILE = "sigma.txt"
ACCEL_MODEL_ERROR_SIGMA = 0.7  # m/s**2


class Observer:
    def __init__(self, ts_control, initial_measurements = MsgSensors()):
        # initialized estimated state message
        self.estimated_state = MsgState()
        # use alpha filters to low pass filter gyros and accels
        # alpha = Ts/(Ts + tau) where tau is the LPF time constant

        ##### TODO #####
        self.lpf_gyro_x = AlphaFilter(alpha=0.75, y0=initial_measurements.gyro_x, sigma=SENSOR.gyro_sigma)
        self.lpf_gyro_y = AlphaFilter(alpha=0.75, y0=initial_measurements.gyro_y, sigma=SENSOR.gyro_sigma)
        self.lpf_gyro_z = AlphaFilter(alpha=0.75, y0=initial_measurements.gyro_z, sigma=SENSOR.gyro_sigma)
        self.lpf_accel_x = AlphaFilter(alpha=0.75, y0=initial_measurements.accel_x, sigma=SENSOR.accel_sigma)
        self.lpf_accel_y = AlphaFilter(alpha=0.75, y0=initial_measurements.accel_y, sigma=SENSOR.accel_sigma)
        self.lpf_accel_z = AlphaFilter(alpha=0.75, y0=initial_measurements.accel_z, sigma=SENSOR.accel_sigma)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.85, y0=initial_measurements.abs_pressure, sigma=SENSOR.abs_pres_sigma)
        self.lpf_diff = AlphaFilter(alpha=0.85, y0=initial_measurements.diff_pressure, sigma=SENSOR.diff_pres_sigma)
        
        self.estimated_state.p = self.lpf_gyro_x.y
        self.estimated_state.sigma_p = self.lpf_gyro_x.sigma 
        self.estimated_state.q = self.lpf_gyro_y.y
        self.estimated_state.sigma_q = self.lpf_gyro_y.sigma
        self.estimated_state.r = self.lpf_gyro_z.y
        self.estimated_state.sigma_r = self.lpf_gyro_z.sigma

        # invert sensor model to get altitude and airspeed
        self.estimated_state.altitude = self.lpf_abs.y / (MAV.rho * MAV.gravity) 
        self.estimated_state.sigma_altitude = self.lpf_abs.sigma / (MAV.rho * MAV.gravity) 
        self.estimated_state.Va = np.sqrt(2. * self.lpf_diff.y / MAV.rho)
        self.estimated_state.sigma_Va = self.lpf_diff.sigma / np.sqrt(2. * MAV.rho * self.lpf_diff.y)

        # ekf for phi, theta, psi
        self.attitude_ekf = EkfAttitude(initial_measurements, self.estimated_state)
        self.estimated_state.phi = self.attitude_ekf.xhat.item(0)
        self.estimated_state.sigma_phi = np.sqrt(self.attitude_ekf.P[0, 0])
        self.estimated_state.theta = self.attitude_ekf.xhat.item(1)
        self.estimated_state.sigma_theta = np.sqrt(self.attitude_ekf.P[1, 1])

        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition(initial_measurements, self.estimated_state)

    def update(self, measurement):
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x)
        self.estimated_state.sigma_p = self.lpf_gyro_x.sigma
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.sigma_q = self.lpf_gyro_y.sigma
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)
        self.estimated_state.sigma_r = self.lpf_gyro_z.sigma

        # invert sensor model to get altitude and airspeed
        altitude_p = self.lpf_abs.update(measurement.abs_pressure) / (MAV.rho * MAV.gravity) 
        sigma_altitude_p = self.lpf_abs.sigma / (MAV.rho * MAV.gravity) 
        self.estimated_state.Va = np.sqrt(2. * self.lpf_diff.update(measurement.diff_pressure) / MAV.rho)
        self.estimated_state.sigma_Va = self.lpf_diff.sigma / np.sqrt(2. * MAV.rho * self.lpf_diff.y)

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(measurement, self.estimated_state)

        # estimate pn, pe, pd, Vg, chi, gamma, wn, we, wd, psi, gamma_a
        self.position_ekf.update(measurement, self.estimated_state)
        
        # fuse altitude estimates
        altitude_gps = self.estimated_state.altitude
        sigma_altitude_gps = self.estimated_state.sigma_altitude
        K = sigma_altitude_p**2 / (sigma_altitude_p**2 + sigma_altitude_gps**2)
        self.estimated_state.altitude = altitude_p + K * (altitude_gps - altitude_p) 
        self.estimated_state.sigma_altitude = (1. - K) * sigma_altitude_p
        
        alpha_beta = estimate_wind_quantities(self.position_ekf.xhat, measurement, self.estimated_state)
        C_wind = jacobian(estimate_wind_quantities, self.position_ekf.xhat, measurement, self.estimated_state)
        P_wind = C_wind @ self.position_ekf.P @ C_wind.T
        var_alpha, var_beta = np.diag(P_wind)
        self.estimated_state.alpha = alpha_beta.item(0)
        self.estimated_state.sigma_alpha = np.sqrt(var_alpha)
        self.estimated_state.beta = alpha_beta.item(1)
        self.estimated_state.sigma_beta = np.sqrt(var_beta)
        # not estimating these
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0, sigma=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition
        self.sigma0 = sigma
        self.sigma = sigma
        self._k = 0

    def update(self, u):
        ##### TODO #####
        self._k += 1
        y_old = self.y
        self.y = self.alpha * y_old + (1. - self.alpha) * u
        tmp = self.alpha**(2*self._k)
        self.sigma = self.sigma0 * np.sqrt( tmp + ( (1. - self.alpha) * (1. - tmp) / (1. + self.alpha) ) ) 
        return self.y

def calculate_heading_from_mag(measurement, state, psi_init=0.):
    mb_x, mb_y, mb_z = measurement.mag_x, measurement.mag_y, measurement.mag_z
    phi, theta = state.phi, state.theta

    # estimate heading
    # get magnetic north in the inertial frame
    m_i = SENSOR.R_m2i @ np.array([[1.], [0.], [0.]])
    m_b_hat = np.array([[mb_x], [mb_y], [mb_z]])
    # R_v2^v1 = (R_v1^v2).T from pg. 15 of book
    Rth = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]]) 
    # R_b^v2 = (R_v2^b).T from pg. 15 of book
    Rph = np.array([[1., 0., 0.], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]]) 
    m_v1_hat = Rth @ Rph @ m_b_hat
    
    # finding the root is equivalent to solving a minimization problem for the sum of the squared difference over the first two entries (which both include psi-dependent terms)
    def fn(x, m_i, rhs):
        _psi = x[0]
        # R_i^v1
        Rps = np.array([[np.cos(_psi), np.sin(_psi), 0.], [-np.sin(_psi), np.cos(_psi), 0.], [0., 0., 1.]])
        m_v1 = Rps @ m_i
        return (m_v1[0, 0] - rhs[0, 0])**2 + (m_v1[1, 0] - rhs[1, 0])**2
    
    heading_soln = minimize(fn, psi_init, args=(m_i, m_v1_hat))

    psi = None
    if heading_soln.success:
        # only include the heading estimate if the optimization was successful
        psi = heading_soln.x[0]
    return psi


class EkfAttitude:
    # implement continuous-discrete EKF to estimate roll, pitch, and heading angles
    def __init__(self, measurement, state):
        ##### TODO #####
        # select process noise for post-integration step
        # trust the model; it matches the EOM in the simulation
        self.Q = np.diag([np.radians(0.005)**2, np.radians(0.005)**2, np.radians(0.005)**2])
        self.Q_scaling = 1.
        # R_accel has to account for assumptions in the model, plus the noise on state inputs, plus the actual measurement noise
        self.R_accel = np.diag([
            ACCEL_MODEL_ERROR_SIGMA**2 + SENSOR.accel_sigma**2,
            ACCEL_MODEL_ERROR_SIGMA**2 + SENSOR.accel_sigma**2,
            ACCEL_MODEL_ERROR_SIGMA**2 + SENSOR.accel_sigma**2
        ])
        
        self.R_mag = np.diag([SENSOR.mag_heading_sigma**2])
        self.N = 1  # number of prediction step per sample
        p, q, r, Va = state.p, state.q, state.r, state.Va

        theta = np.arcsin( measurement.accel_x / (q * Va + MAV.gravity) ) 
        phi = np.arcsin( (r * Va * np.cos(theta) - p * Va * np.sin(theta) - measurement.accel_y) / (MAV.gravity * np.cos(theta)) ) 
        psi_meas = calculate_heading_from_mag(measurement, state)
        if psi_meas is None:
            psi = 0.
            sigma_psi = np.radians(60)
        else:
            psi = psi_meas
            sigma_psi = np.radians(10./3)
        self.xhat = np.array([[phi], [theta], [psi]])
        self.P = np.diag([np.radians(10./3)**2, np.radians(10./3)**2, sigma_psi**2])
        self.Ts = SIM.ts_control/self.N
        
        # reject measurements further than 2-sigma from the pseudomeasurement:
        # this condition happens when the mahalanobis distance of the measurement residual is greater than the gate threshold
        self.gate_threshold_accel = stats.chi2.isf(q=0.05, df=3)
        self.gate_threshold_full = stats.chi2.isf(q=0.05, df=4)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0)
        state.sigma_phi = np.sqrt(self.P[0, 0])
        state.theta = self.xhat.item(1)
        state.sigma_theta = np.sqrt(self.P[1, 1])
        state.psi = self.xhat.item(2)
        state.sigma_psi = np.sqrt(self.P[2, 2])

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        ##### TODO #####
        p, q, r = state.p, state.q, state.r
        phi, theta = [x.item(i) for i in range(2)] 
        f_ = np.array([
            [p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r],
            [np.cos(phi) * q - np.sin(phi) * r],
            [np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r]
        ])
        return f_

    def f_s(self, x, measurement, state):
        # invert: x<->state
        p, q, r = [x.item(i) for i in range(3)]
        phi, theta = [state.item(i) for i in range(2)]
        f_ = np.array([
            [p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r],
            [np.cos(phi) * q - np.sin(phi) * r],
            [np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r]
        ])
        return f_

    def h_accel(self, x, measurement, state):
        # measurement model y
        ##### TODO #####
        p, q, r, Va = state.p, state.q, state.r, state.Va
        phi, theta = [x.item(i) for i in range(2)]
        
        h_ = np.array([
            [q * Va * np.sin(theta) + MAV.gravity * np.sin(theta)],  # x-accel
            [r * Va * np.cos(theta) - p * Va * np.sin(theta) - MAV.gravity * np.cos(theta) * np.sin(phi)],  # y-accel
            [-q * Va * np.cos(theta) - MAV.gravity * np.cos(theta) * np.cos(phi)],  # z-accel
        ])  
        return h_
    
    def h_accel_s(self, x, measurement, state):
        # invert: x<->state
        p, q, r, Va = [x.item(i) for i in range(4)]
        phi, theta = [state.item(i) for i in range(2)]
        h_ = np.array([
            [q * Va * np.sin(theta) + MAV.gravity * np.sin(theta)],  # x-accel
            [r * Va * np.cos(theta) - p * Va * np.sin(theta) - MAV.gravity * np.cos(theta) * np.sin(phi)],  # y-accel
            [-q * Va * np.cos(theta) - MAV.gravity * np.cos(theta) * np.cos(phi)],  # z-accel
        ])  
        return h_
    
    def h_mag(self, x, measurement, state):
        # measurement model y
        h_ = np.array([
            [x.item(2)]  # psi
        ])  
        return h_
    
    
    def propagate_model(self, measurement, state):
        # model propagation
        Tp = self.Ts / self.N
        for _ in range(0, self.N):
            A = jacobian(self.f, self.xhat, measurement, state)
            self.xhat += self.f(self.xhat, measurement, state) * Tp
            
            # account for error in p, q, r here
            x_s = np.array([[state.p], [state.q], [state.r], [state.Va]])
            J_s = jacobian(self.f_s, x_s, measurement, self.xhat)
            Q_s = J_s @ np.diag([state.sigma_p**2, state.sigma_q**2, state.sigma_r**2, state.sigma_Va**2]) @ J_s.T

            # total process noise = model error + "state" members  errors
            Q = self.Q_scaling * (self.Q + Q_s)
            self.P += (A @ self.P + self.P @ A.T + Q) * Tp

    def measurement_update(self, measurement, state):
        xhat = np.copy(self.xhat)
        P = np.copy(self.P)
        # measurement updates
        pseudo_meas_accel = self.h_accel(xhat, measurement, state)
        C_accel = jacobian(self.h_accel, xhat, measurement, state)
        y_accel = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T

        # account for p, q, r and Va errors
        x_accel_s = np.array([[state.p], [state.q], [state.r], [state.Va]])
        J_accel = jacobian(self.h_accel_s, x_accel_s, measurement, self.xhat)
        R_accel = self.R_accel + J_accel @ np.diag([state.sigma_p**2, state.sigma_q**2, state.sigma_r**2, state.sigma_Va**2]) @ J_accel.T
        
        # mag
        psi_meas = calculate_heading_from_mag(measurement, state, psi_init=self.xhat.item(2))
        if psi_meas is not None:
            pseudo_meas_mag = self.h_mag(xhat, measurement, state)
            C_mag = jacobian(self.h_mag, xhat, measurement, state)
            y_mag = np.array([[wrap(psi_meas, pseudo_meas_mag[0, 0])]]).T

            pseudo_meas = np.concatenate((pseudo_meas_accel, pseudo_meas_mag))
            y = np.concatenate((y_accel, y_mag))
            C = np.concatenate((C_accel, C_mag))
            # heading is measured directly, so no transform of the standard deviation is needed
            R = np.block([[R_accel, np.zeros((3, 1))], [np.zeros((1, 3)), self.R_mag]])
            gate_threshold = self.gate_threshold_full
        else:
            pseudo_meas = pseudo_meas_accel
            y = y_accel
            C = C_accel
            R = R_accel
            gate_threshold = self.gate_threshold_accel

        S = C @ P @ C.T + R
        S_inv = np.linalg.inv(S)
        residual = y - pseudo_meas
        mahalonobis_distance_sq = float(residual.T @ S_inv @ residual)

        if mahalonobis_distance_sq < gate_threshold:
            L = P @ C.T @ S_inv
            self.xhat = xhat + L @ residual
            self.P = (np.eye(3) - L @ C) @ P
            self.Q_scaling = 100*(mahalonobis_distance_sq / gate_threshold) 
        else:
            self.Q_scaling = 1000 * (mahalonobis_distance_sq / gate_threshold)

def estimate_wind_quantities(x, measurement, state):
    Vg, chi, gamma, wn, we, wd = [x.item(i) for i in range(3, 9)]
    wind_i = np.array([
        [wn],
        [we],
        [wd]
    ])

    v_i = np.array([
        [Vg * np.cos(chi) * np.cos(gamma)],
        [Vg * np.sin(chi) * np.cos(gamma)],
        [-Vg * np.sin(gamma)],
    ])

    R_b2i = Euler2Rotation(state.phi, state.theta, state.psi)

    vb_r = R_b2i.T @ (v_i - wind_i)
    u_r, v_r, w_r = [vb_r.item(i) for i in range(3)]
    
    alpha = np.arctan2(w_r, u_r)
    beta = np.arcsin(v_r / state.Va)

    return np.array([[alpha], [beta]])


class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, pd, Vg, chi, gamma, wn, we, wd
    def __init__(self, measurement, state):
        self.xhat = np.array([
            [measurement.gps_n],
            [measurement.gps_e],
            [measurement.gps_h],
            [measurement.gps_Vg],
            [measurement.gps_course],
            [measurement.gps_gamma],
            [0.0],
            [0.0],
            [0.0]
        ])
        
        self.P = np.diag([10**2, 10**2, 10**2, 10**2, np.radians(10./3)**2, np.radians(10./3)**2, (MAX_WIND/3)**2, (MAX_WIND/3)**2, (MAX_WIND/3)**2])
        
        self.N = 1  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        
        # the model introduces uncertainty from turning rate model
        # errors from state will be added later
        self.Q = np.diag([
            (0.005)**2,  # pn
            (0.005)**2,  # pe
            (0.005)**2,  # h
            (0.005)**2,  # Vg
            np.radians(0.005)**2,  # chi
            np.radians(0.005)**2,  # gamma
            (0.1)**2,  # wn  # XXX unsure of wind dynamics, so pad the process noise accordingly
            (0.1)**2,  # we
            (0.1)**2,  # wd
        ])
        self.Q_scaling = 1.
 
        self.gps_n_old = 0
        self.gps_e_old = 0
        self.gps_h_old = 0
        self.gps_Vg_old = 0
        self.gps_course_old = 0
        self.gps_gamma_old = 0

        self.gps_error_constant = float(np.exp(-SENSOR.gps_k * SENSOR.ts_gps))
        self.gps_count = 0
        
        self.gate_threshold_pseudo = stats.chi2.isf(q=0.05, df=3)
        self.gate_threshold_gps = stats.chi2.isf(q=0.05, df=6)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.sigma_north = np.sqrt(self.P[0, 0])
        state.east = self.xhat.item(1)
        state.sigma_east = np.sqrt(self.P[1, 1])
        state.altitude = -self.xhat.item(2)
        state.sigma_altitude = np.sqrt(self.P[2, 2])
        state.Vg = self.xhat.item(3)
        state.sigma_Vg = np.sqrt(self.P[3, 3])
        state.chi = self.xhat.item(4)
        state.sigma_chi = np.sqrt(self.P[4, 4])
        state.gamma = self.xhat.item(5)
        state.sigma_gamma = np.sqrt(self.P[5, 5])
        state.wn = self.xhat.item(6)
        state.sigma_wn = np.sqrt(self.P[6, 6])
        state.we = self.xhat.item(7)
        state.sigma_we = np.sqrt(self.P[7, 7])
        state.wd = self.xhat.item(8)
        state.sigma_wd = np.sqrt(self.P[8, 8])

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        Vg, chi, gamma = [x.item(i) for i in range(3, 6)]
        
        # derivatives from state
        phi, theta, psi, q, r, Va = state.phi, state.theta, state.psi, state.q, state.r, state.Va
        psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r

        # derivatives from x
        # assumptions:
        # 1) Va is constant
        # 2) wind is constant
        Vn = Vg * np.cos(chi) * np.cos(gamma)  # == Va * np.cos(psi) * np.cos(theta - alpha) + wn
        Vn_dot = -Va * psi_dot * np.sin(psi)
        Ve = Vg * np.sin(chi) * np.cos(gamma)  # == Va * np.sin(psi) * np.cos(theta - alpha) + we
        Ve_dot = Va * psi_dot * np.cos(psi)
        Vd = -Vg * np.sin(gamma)  # -Va * np.sin(theta - alpha) + wd 
        #Vd_dot = 0  # gamma_a_dot = 0
        Vg_dot = (Vn * Vn_dot + Ve * Ve_dot) / Vg
        chi_dot = (Ve * Vn_dot - Ve_dot * Vn) / (Vn**2 + Ve**2)
        gamma_dot = Vd * Vg_dot / ( Vg**2 * np.sqrt(1. - Vd**2 / Vg**2) )
        f_ = np.array([
            [Vn],
            [Ve],
            [Vd],
            [Vg_dot],
            [chi_dot],
            [gamma_dot],
            [0],
            [0],
            [0],
        ])
        return f_

    def f_s(self, x, measurement, state):
        # invert: x<->state
        Vg, chi, gamma = [state.item(i) for i in range(3, 6)]
        
        # derivatives from state
        # gamma_a_dot = theta_dot (assumes alpha_dot = 0)
        phi, theta, psi, q, r, Va, alpha = [x.item(i) for i in range(7)]
        psi_dot = np.sin(phi) / np.cos(theta) * q + np.cos(phi) / np.cos(theta) * r

        # derivatives from x
        # assumptions:
        # 1) Va is constant
        # 2) wind is constant
        Vn = Vg * np.cos(chi) * np.cos(gamma)  # == Va * np.cos(psi) * np.cos(theta - alpha) + wn
        Vn_dot = -Va * psi_dot * np.sin(psi) * np.cos(theta - alpha)  # theta_dot = alpha_dot
        Ve = Vg * np.sin(chi) * np.cos(gamma)  # == Va * np.sin(psi) * np.cos(theta - alpha) + we
        Ve_dot = Va * psi_dot * np.cos(psi) * np.cos(theta - alpha)
        Vd = -Vg * np.sin(gamma)  # -Va * np.sin(theta - alpha) + wd 
        #Vd_dot = 0  # gamma_a_dot = 0
        Vg_dot = (Vn * Vn_dot + Ve * Ve_dot) / Vg
        chi_dot = (Ve * Vn_dot - Ve_dot * Vn) / (Vn**2 + Ve**2)
        gamma_dot = Vd * Vg_dot / ( Vg**2 * np.sqrt(1. - Vd**2 / Vg**2) )
        f_ = np.array([
            [Vn],
            [Ve],
            [Vd],
            [Vg_dot],
            [chi_dot],
            [gamma_dot],
            [0],
            [0],
            [0],
        ])
        return f_

    def h_gps(self, x, measurement, state):
        # measurement model for gps measurements
        n, e, d, Vg, chi, gamma = [x.item(i) for i in range(6)]
        h_ = np.array([
            [n],  # pn
            [e],  # pe
            [-d],  # h
            [Vg],  # Vg
            [chi],  # chi
            [gamma],  # gamma
        ])
        return h_

    def estimate_gps_uncertainty(self, x):
        Vg, chi, gamma = [x.item(i) for i in range(3, 6)]
        a = self.gps_error_constant
        m = (1. - a**(self.gps_count+1)) / (1. - a) 
        Vn = Vg * np.cos(chi) * np.cos(gamma)
        Ve = Vg * np.sin(chi) * np.cos(gamma)
        return np.diag([
            (m+1)*SENSOR.gps_n_sigma**2,
            (m+1)*SENSOR.gps_e_sigma**2,
            (m+1)*SENSOR.gps_h_sigma**2,
            SENSOR.gps_Vg_sigma**2,
            SENSOR.gps_Vg_sigma**2 / (Vn**2 + Ve**2),
            SENSOR.gps_Vg_sigma**2 / Vg**2
        ])

    def h_pseudo(self, x, measurement, state):
        # measurement model for wind triangle pseudo measurement
        Vg, chi, gamma, wn, we, wd = [x.item(i) for i in range(3, 9)]
        psi, Va = state.psi, state.Va
        Vn_lhs = Vg * np.cos(chi) * np.cos(gamma)
        Ve_lhs = Vg * np.sin(chi) * np.cos(gamma)
        Vd_lhs = -Vg * np.sin(gamma)
        Vn_rhs = Va * np.cos(psi) + wn  # XXX need to account for theta = alpha assumption in this term
        Ve_rhs = Va * np.sin(psi) + we
        Vd_rhs = wd
        h_ = np.array([
            [Vn_lhs - Vn_rhs],  # wind triangle x
            [Ve_lhs - Ve_rhs],  # wind triangle y
            [Vd_lhs - Vd_rhs],  # wind triangle z
        ])
        return h_
    
    def h_pseudo_s(self, x, measurement, state):
        # invert x <--> state
        # measurement model for wind triangle pseudo measurement
        Vg, chi, gamma, wn, we, wd = [state.item(i) for i in range(3, 9)]
        psi, Va, theta, alpha = [x.item(i) for i in range(4)]
        gamma_a = theta - alpha
        Vn_lhs = Vg * np.cos(chi) * np.cos(gamma)
        Ve_lhs = Vg * np.sin(chi) * np.cos(gamma)
        Vd_lhs = -Vg * np.sin(gamma)
        Vn_rhs = Va * np.cos(psi) * np.cos(gamma_a) + wn
        Ve_rhs = Va * np.sin(psi) * np.cos(gamma_a) + we
        Vd_rhs = -Va * np.sin(gamma_a) + wd
        h_ = np.array([
            [Vn_lhs - Vn_rhs],  # wind triangle x
            [Ve_lhs - Ve_rhs],  # wind triangle y
            [Vd_lhs - Vd_rhs],  # wind triangle z
        ])
        return h_
    
    
    def h_alt(self, x, measurement, state):
        # measurement model y
        h_ = np.array([
            [-x.item(2)]  # altitude
        ])  
        return h_
    
    def propagate_model(self, measurement, state):
        # model propagation
        Tp = self.Ts / self.N
        for i in range(0, self.N):
            A = jacobian(self.f, self.xhat, measurement, state)
            self.xhat += self.f(self.xhat, measurement, state) * Tp
            
            # account for error in p, q, r here
            x_s = np.array([[state.phi], [state.theta], [state.psi], [state.q], [state.r], [state.Va], [state.theta]])
            J_s = jacobian(self.f_s, x_s, measurement, self.xhat)
            Q_s = J_s @ np.diag([state.sigma_phi**2, state.sigma_theta**2, state.sigma_psi**2, state.sigma_q**2, state.sigma_r**2, state.sigma_Va**2, state.sigma_theta**2]) @ J_s.T

            # total process noise = model error + "state" members  errors
            Q = self.Q_scaling * (self.Q + Q_s)
            self.P += (A @ self.P + self.P @ A.T + Q) * Tp

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        xhat = np.copy(self.xhat)
        P = np.copy(self.P)
        pseudo_meas_pseudo = self.h_pseudo(xhat, measurement, state)
        C_pseudo = jacobian(self.h_pseudo, xhat, measurement, state)
        y_pseudo = np.array([[0., 0., 0.]]).T

        # account for error in psi, va, theta, and alpha
        x_s = np.array([[state.psi, state.Va, state.theta, state.theta]]).T
        J_s = jacobian(self.h_pseudo_s, x_s, measurement, xhat)
        R_s = np.diag(np.array([state.sigma_psi**2, state.sigma_Va**2, state.sigma_theta**2, state.sigma_theta**2]))
        R_pseudo = J_s @ R_s @ J_s.T

        S_pseudo = C_pseudo @ P @ C_pseudo.T + R_pseudo
        S_pseudo_inv = np.linalg.inv(S_pseudo)
        residual_pseudo = y_pseudo - pseudo_meas_pseudo
        mahalonobis_distance_pseudo = float(residual_pseudo.T @ S_pseudo_inv @ residual_pseudo)
        if mahalonobis_distance_pseudo < self.gate_threshold_pseudo:
            L_pseudo = P @ C_pseudo.T @ S_pseudo_inv
            self.xhat = xhat + L_pseudo @ residual_pseudo
            self.P = (np.eye(9) - L_pseudo @ C_pseudo) @ P
            self.Q_scaling = 100 * (mahalonobis_distance_pseudo / self.gate_threshold_pseudo)
        else:
            self.Q_scaling = 5000 * (mahalonobis_distance_pseudo / self.gate_threshold_pseudo)

        # only update GPS when one of the signals changes
        # gps measurement model directly measures state, so no transform needed
        xhat = np.copy(self.xhat)
        P = np.copy(self.P)
        if not np.isclose(measurement.gps_n, self.gps_n_old) \
            or not np.isclose(measurement.gps_e, self.gps_e_old) \
            or not np.isclose(measurement.gps_h, self.gps_h_old) \
            or not np.isclose(measurement.gps_Vg, self.gps_Vg_old) \
            or not np.isclose(measurement.gps_course, self.gps_course_old) \
            or not np.isclose(measurement.gps_gamma, self.gps_gamma_old):

            h_gps = self.h_gps(self.xhat, measurement, state)
            C_gps = jacobian(self.h_gps, self.xhat, measurement, state)
            y_chi = wrap(measurement.gps_course, h_gps[4, 0])
            y_gamma = wrap(measurement.gps_gamma, h_gps[5, 0])
            y_gps = np.array([[measurement.gps_n,
                               measurement.gps_e,
                               measurement.gps_h,
                               measurement.gps_Vg,
                               y_chi,
                               y_gamma]]).T

            R_gps = self.estimate_gps_uncertainty(xhat)
            S_gps = C_gps @ P @ C_gps.T + R_gps
            S_gps_inv = np.linalg.inv(S_gps)
            residual_gps = y_gps - h_gps
            mahalonobis_distance_gps = float(residual_gps.T @ S_gps_inv @ residual_gps)
            if mahalonobis_distance_gps < 1000:  # don't gate gps
                L_gps = P @ C_gps.T @ S_gps_inv
                self.xhat = xhat + L_gps @ residual_gps
                self.P = (np.eye(9) - L_gps @ C_gps) @ P
                self.Q_scaling = 500 * (mahalonobis_distance_gps / self.gate_threshold_gps)
            else:
                self.Q_scaling = 5000 * (mahalonobis_distance_gps / self.gate_threshold_gps)

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_h_old = measurement.gps_h
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course
            self.gps_gamma_old = measurement.gps_gamma

            self.gps_count += 1


def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J

