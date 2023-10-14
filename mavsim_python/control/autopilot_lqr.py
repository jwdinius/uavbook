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
from tools.wrap import wrap
import design_projects.chap05.model_coef as M
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from importlib import import_module


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
        self.integratorAltitude = 0
        self.integratorAirspeed = 0
        self.errorSideslipD1 = 0  # == error at last step; discrete representation of an integrator requires it ("D1" means "delay one")
        self.errorCourseD1 = 0
        self.errorAltitudeD1 = 0
        self.errorAirspeedD1 = 0
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
        CLon = array([[0, 0., 0., 0., 1.],
                   [1., 0., 0., 0., 0.]])
        CrLon = concatenate((-CLon, zeros((2, 2))), axis=1)
        AAlon = concatenate((
                    concatenate((M.A_lon_w_alpha, zeros((5,2))), axis=1),
                    CrLon),
                    axis=0)
        BBlon = concatenate((M.B_lon_w_alpha, zeros((2, 2))), axis=0)
        Qlon = diag([
            self.AP.max_delta_airspeed**(-2),
            self.AP.max_delta_alpha**(-2),
            self.AP.max_delta_q**(-2),
            self.AP.max_delta_theta**(-2),
            self.AP.max_delta_altitude**(-2),
            self.AP.max_altitude_int**(-2),
            self.AP.max_airspeed_int**(-2)])
        Rlon = diag([
            self.AP.max_elevator**(-2),
            1**(-2)]) # e, t
        Plon = solve_continuous_are(AAlon, BBlon, Qlon, Rlon)
        self.Klon_aug = inv(Rlon) @ BBlon.T @ Plon
        self.Klon = self.Klon_aug[:, :5] 
        self.Klon_i = self.Klon_aug[:, 5:] 
        self.Klon_r = -inv(CLon @ inv(M.A_lon_w_alpha - M.B_lon_w_alpha @ self.Klon) @ M.B_lon_w_alpha)  # track reference inputs: h_c and Va_c
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
        # NOTE: `u` is not part of the state vector provided, so it must be reconstructed from airspeed, alpha, and beta (which is assumed to be ~0)
        u = state.Va * np.cos(state.alpha) * np.cos(state.beta)
        trim_u = self._trim_state.Va * np.cos(self._trim_state.alpha) * np.cos(self._trim_state.beta)
        delta_u = u - trim_u 
        delta_alpha = state.alpha - wrap(self._trim_state.alpha, state.alpha)
        delta_q = state.q - self._trim_state.q
        delta_theta = state.theta - wrap(self._trim_state.theta, state.theta)
        delta_altitude = state.altitude - self._trim_state.altitude
        altitude_error = cmd.altitude_command - state.altitude
        # we _want_ alpha and beta to be near zero, so u ~ airspeed
        airspeed_error = cmd.airspeed_command - u
        self.integratorAltitude += 0.5 * self.Ts * (altitude_error + self.errorAltitudeD1)
        self.integratorAirspeed += 0.5 * self.Ts * (airspeed_error + self.errorAirspeedD1)
        
        # states
        x_lon = np.array([
            [delta_u],
            [delta_alpha],
            [delta_q],
            [delta_theta],
            [delta_altitude]
        ])

        # augmented state (integrals)
        z_lon = np.array([
            [self.integratorAltitude],
            [self.integratorAirspeed]
        ])

        # reference input (which accounts for delta from trim)
        y_d_lon = np.array([
            [cmd.altitude_command - self._trim_state.altitude], 
            [cmd.airspeed_command - trim_u]
        ])

        u_lon_star = np.array([
            [self._trim_input.elevator],
            [self._trim_input.throttle]
        ])

        # compute unsatured input, which is the sum of 4 terms:
        # 1) trim input
        # 2) reference tracking input: Kr @ y_d
        # 3) state feedback: -K @ x
        # 4) augmented state (integral) feedback: -Ki @ z
        u_lon_unsat = u_lon_star + self.Klon_r @ y_d_lon - self.Klon @ x_lon  - self.Klon_i @ z_lon

        # apply control saturation
        u_lon_sat = np.array([
            [saturate(u_lon_unsat.item(0), -self.AP.max_elevator, self.AP.max_elevator)],
            [saturate(u_lon_unsat.item(1), 0, 1)]
        ])

        # apply antiwindup correction
        delta_z_lon = integratorAntiWindup(self.Klon_i, z_lon, u_lon_unsat, u_lon_sat)
        self.integratorAltitude += delta_z_lon.item(0)
        self.integratorAirspeed += delta_z_lon.item(1)

        # write error terms for next pass
        self.errorSideslipD1 = sideslip_error
        self.errorCourseD1 = course_error
        self.errorAltitudeD1 = altitude_error
        self.errorAirspeedD1 = airspeed_error

        # construct control outputs and commanded states
        delta = MsgDelta(elevator=u_lon_sat.item(0),
                         aileron=u_lat_sat.item(0),
                         rudder=u_lat_sat.item(1),
                         throttle=u_lon_sat.item(1))
        self.commanded_state.altitude = cmd.altitude_command 
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = 0  # phi_c
        self.commanded_state.theta = 0  # theta_c
        self.commanded_state.chi = cmd.course_command
        
        return delta, self.commanded_state

