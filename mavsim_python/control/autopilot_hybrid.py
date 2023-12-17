"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/10/22 - RWB
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.environ["UAVBOOK_HOME"])
import numpy as np
from numpy import array, sin, cos, radians, concatenate, zeros, diag
from scipy.linalg import solve_continuous_are, inv
from control.create_throttle_lut import calculate_motor_thrust
from control.pi_control import PIControl
from control.pd_control_with_rate import PDControlWithRate
from tools.wrap import wrap
import design_projects.chap05.model_coef as M
import parameters.aerosonde_parameters as MAV
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from scipy.interpolate import RegularGridInterpolator
from importlib import import_module

MAX_LUT_ERROR_SQ=5e-2


def obj_fn(_delta_t, _airspeed, _T_des):
    return (calculate_motor_thrust(_delta_t, _airspeed) - _T_des)**2

def saturate(inp, low_limit, up_limit):
    if inp <= low_limit:
        output = low_limit
    elif inp >= up_limit:
        output = up_limit
    else:
        output = inp
    return output

def integratorAntiWindup(Ki, z_n, u_unsat, u_sat):
    '''
    z_n == z^-
    z_p == z^+
    z_p = z_n + delta_z

    return the delta_z increment to apply to the integrator to prevent windup (ensure command is at most at saturation)
    '''
    if np.allclose(u_unsat, u_sat):
        # no correction needed
        return np.zeros_like(z_n)

    return np.linalg.solve(Ki, u_unsat - u_sat)

class Autopilot:
    def __init__(self, ts_control, use_truth=False):
        if use_truth:
            self.AP = import_module("parameters.control_parameters")
        else:
            self.AP = import_module("parameters.control_parameters_estimator")
        self.Ts = ts_control
        # initialize integrators and delay variables
        self.integratorSideslip = 0
        self.integratorCourse = 0
        self.errorSideslipD1 = 0  # == error at last step; discrete representation of an integrator requires it ("D1" means "delay one")
        self.errorCourseD1 = 0
        # compute LQR gains
        
        #### TODO ######
        '''Lateral autopilot
        Objectives: drive vehicle to desired course (heading) angle and regulate beta (to 0)
        '''
        # augmented state vector is [[beta], [p], [r], [phi], [psi (~chi)], [beta integral], [psi (~chi) integral]].T
        CLat = array([[1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1.]])
        CrLat = concatenate((-CLat, zeros((2, 2))), axis=1)
        AAlat = concatenate((
                    concatenate((M.A_lat_w_beta, zeros((5,2))), axis=1),
                    CrLat),
                    axis=0)
        BBlat = concatenate((M.B_lat_w_beta, zeros((2,2))), axis=0)
        # use Bryson's rule for tuning the diagonal terms of Q, R:
        # i.e., take the maximum acceptable value of each state/input term
        # and square it, then invert it
        Qlat = diag([
            self.AP.max_delta_sideslip**(-2),
            self.AP.max_delta_p**(-2),
            self.AP.max_delta_r**(-2),
            self.AP.max_delta_phi**(-2),
            self.AP.max_delta_chi**(-2),
            self.AP.max_sideslip_int**(-2),
            self.AP.max_chi_int**(-2)]) # beta, p, r, phi, chi, intBeta, intChi, gains from the notes
        Rlat = diag([
            self.AP.max_aileron**(-2),
            self.AP.max_rudder**(-2)]) # a, r  TODO: this max should account for trim input
        Plat = solve_continuous_are(AAlat, BBlat, Qlat, Rlat)
        self.Klat_aug = inv(Rlat) @ BBlat.T @ Plat
        self.Klat = self.Klat_aug[:, :5] 
        self.Klat_i = self.Klat_aug[:, 5:]
        self.Klat_r = -inv(CLat @ inv(M.A_lat_w_beta - M.B_lat_w_beta @ self.Klat) @ M.B_lat_w_beta)  # track reference inputs: beta_c (= 0), and chi_c using non-augmented state matrices
        
        '''Longitudinal autopilot
        Objectives: drive vehicle to desired altitude and airspeed
        '''
        self.k_T = self.AP.k_T_tecs 
        self.k_D = self.AP.k_D_tecs 
        self.k_Va = self.AP.k_Va_tecs 
        self.k_h = self.AP.k_h_tecs
        self.pitch_from_elevator = PDControlWithRate(
                        kp=self.AP.pitch_kp_tecs,
                        kd=self.AP.pitch_kd_tecs,
                        limit=self.AP.max_elevator)
        self.throttle_correction_from_airspeed = PIControl(
                        kp=self.AP.throttle_correction_kp_tecs,
                        ki=self.AP.throttle_correction_ki_tecs,
                        Ts=ts_control,
                        limit=self.AP.throttle_correction_limit)
        self.pitch_correction_from_altitude = PIControl(
                        kp=self.AP.altitude_correction_kp_tecs,
                        ki=self.AP.altitude_correction_ki_tecs,
                        Ts=ts_control,
                        limit=self.AP.altitude_correction_limit)

        
        with open(os.environ["UAVBOOK_HOME"] + "/control/tecs_thrust_lut.txt", "r") as f:
            lines = f.readlines()
        self.thrust_min, self.thrust_max, thrust_size = [float(x) for x in lines[0].split(",")]
        thrust_x = np.linspace(self.thrust_min, self.thrust_max, int(thrust_size))
        airspeed_min, airspeed_max, airspeed_size = [float(x) for x in lines[1].split(",")]
        airspeed_x = np.linspace(airspeed_min, airspeed_max, int(airspeed_size))
        X, _ = np.meshgrid(thrust_x, airspeed_x, indexing="ij")
        Z = np.zeros_like(X)
        for line in lines[2:]:
            i, j, _, _, throttle = [float(x) for x in line.split(",")]
            Z[int(j), int(i)] = throttle
        self.throttle_lut = RegularGridInterpolator((thrust_x, airspeed_x), Z, method='cubic')

        self.delta_t_d1 = 0.5
        self.commanded_state = MsgState()

    def set_trim_input(self, trim_input):
        self._trim_input = trim_input

    def set_trim_state(self, trim_state):
        self._trim_state = trim_state

    def update(self, cmd, state):
        # lateral autopilot
        delta_sideslip = state.beta - wrap(self._trim_state.beta, state.beta)
        delta_p = state.p - self._trim_state.p
        delta_r = state.r - self._trim_state.r
        delta_phi = state.phi - wrap(self._trim_state.phi, state.phi)
        delta_chi = state.chi - wrap(self._trim_state.chi, state.chi)
        sideslip_command = 0
        sideslip_error = sideslip_command - wrap(state.beta, sideslip_command)
        course_error = cmd.course_command - wrap(state.chi, cmd.course_command)
        self.integratorSideslip += 0.5 * self.Ts * (sideslip_error + self.errorSideslipD1)
        self.integratorCourse += 0.5 * self.Ts * (course_error + self.errorCourseD1)
        
        # states
        x_lat = np.array([
            [delta_sideslip],
            [delta_p],
            [delta_r],
            [delta_phi],
            [delta_chi]
        ])

        # augmented state (integrals)
        z_lat = np.array([
            [self.integratorSideslip],
            [self.integratorCourse]
        ])
        
        # reference input (which accounts for delta from trim)
        ## wrap sideslip and heading commands so that they are within +/- pi of current sideslip and heading, respectively
        y_d_lat = np.array([
            [wrap(sideslip_command - wrap(self._trim_state.beta, sideslip_command), state.beta)], 
            [wrap(cmd.course_command - wrap(self._trim_state.chi, cmd.course_command), state.chi)]
        ])

        u_lat_star = np.array([
            [self._trim_input.aileron],
            [self._trim_input.rudder]
        ])

        # compute unsatured input, which is the sum of 4 terms:
        # 1) trim input
        # 2) reference tracking input: Kr @ y_d
        # 3) state feedback: -K @ x
        # 4) augmented state (integral) feedback: -Ki @ z
        u_lat_unsat = u_lat_star + self.Klat_r @ y_d_lat - self.Klat @ x_lat - self.Klat_i @ z_lat

        # apply control saturation
        u_lat_sat = np.array([
            [saturate(u_lat_unsat.item(0), -self.AP.max_aileron, self.AP.max_aileron)],
            [saturate(u_lat_unsat.item(1), -self.AP.max_rudder, self.AP.max_rudder)]
        ])

        # apply antiwindup correction
        delta_z_lat = integratorAntiWindup(self.Klat_i, z_lat, u_lat_unsat, u_lat_sat)
        self.integratorSideslip += delta_z_lat.item(0)
        self.integratorCourse += delta_z_lat.item(1)

        # longitudinal autopilot
        # compute total energy error
        E_K_des = 0.5 * MAV.mass * cmd.airspeed_command**2
        E_P_des = MAV.mass * MAV.gravity * cmd.altitude_command
        E_K = 0.5 * MAV.mass * state.Va**2
        E_P = MAV.mass * MAV.gravity * state.altitude
        # compute desired energy derivatives (assuming first order lag on airspeed and altitude errors)
        airspeed_error = cmd.airspeed_command - state.Va
        altitude_error = cmd.altitude_command - state.altitude
        Va_des_dot = self.k_Va * airspeed_error
        h_des_dot = self.k_h * altitude_error

        E_K_des_dot = MAV.mass * cmd.airspeed_command * Va_des_dot 
        E_P_des_dot = MAV.mass * MAV.gravity * h_des_dot
        E_T_des_dot = E_K_des_dot + E_P_des_dot
        E_K_tilde = E_K_des - E_K
        E_P_tilde = E_P_des - E_P
        E_T_tilde = E_P_tilde + E_K_tilde
        
        ## thrust
        Tc = state.F_drag + (E_T_des_dot + self.k_T * E_T_tilde) / state.Va
        
        # lookup throttle command
        if Tc < self.thrust_min:
            throttle = 0.
        elif Tc > self.thrust_max:
            throttle = 1.
        else:
            throttle = self.throttle_lut((Tc, state.Va))
            if obj_fn(throttle, state.Va, Tc) > MAX_LUT_ERROR_SQ:
                # lookup has unacceptable error, use previous value
                throttle = self.delta_t_d1

        # deal with steady-state error
        throttle += self.throttle_correction_from_airspeed.update(cmd.airspeed_command, state.Va)

        # saturate
        throttle = saturate(throttle, 0, 1)

        ## pitch angle
        sin_gamma_c = h_des_dot / state.Va + ( (self.k_T - self.k_D) * E_K_tilde + (self.k_T + self.k_D) * E_P_tilde ) / ( 2 * MAV.mass * MAV.gravity * state.Va ) 
            
        sin_gamma_c_sat = saturate(sin_gamma_c, -np.sin(self.AP.max_pitch), np.sin(self.AP.max_pitch))
        # theta_c - alpha = gamma_a
        # - gamma_a is the air mass referenced flight path angle.  When wind velocity
        # - is 0, gamma = gamma_a.
        theta_c = state.alpha + np.arcsin(sin_gamma_c_sat)
        
        # deal with steady-state error
        theta_c += self.pitch_correction_from_altitude.update(cmd.altitude_command, state.altitude)
        elevator = self.pitch_from_elevator.update(theta_c, state.theta, state.q)

        # construct control outputs and commanded states
        self.delta_t_d1 = throttle
        delta = MsgDelta(elevator=elevator,
                         aileron=u_lat_sat.item(0),
                         rudder=u_lat_sat.item(1),
                         throttle=throttle)
        self.commanded_state.altitude = cmd.altitude_command 
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = 0  # phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        
        return delta, self.commanded_state

