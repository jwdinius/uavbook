"""
autopilot block for mavsim_python - Total Energy Control System
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/14/2020 - RWB
"""
import os
import sys
import numpy as np
sys.path.append('..')
import parameters.aerosonde_parameters as MAV
from control.tf_control import TFControl
from control.create_throttle_lut import calculate_motor_thrust
from tools.wrap import wrap
from control.pi_control import PIControl
from control.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from scipy.interpolate import RegularGridInterpolator
from importlib import import_module

MAX_LUT_ERROR_SQ=5e-2


def obj_fn(_delta_t, _airspeed, _T_des):
    return (calculate_motor_thrust(_delta_t, _airspeed) - _T_des)**2

class Autopilot:
    def __init__(self, ts_control, use_truth=False):
        if use_truth:
            self.AP = import_module("parameters.control_parameters")
        else:
            self.AP = import_module("parameters.control_parameters_estimator")
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=self.AP.roll_kp,
                        kd=self.AP.roll_kd,
                        limit=self.AP.max_aileron)
        self.course_from_roll = PIControl(
                        kp=self.AP.course_kp,
                        ki=self.AP.course_ki,
                        Ts=ts_control,
                        limit=self.AP.max_roll)
        self.yaw_damper = TFControl(
                        k=self.AP.yaw_damper_kr,
                        n0=0.0,
                        n1=1.0,
                        d0=self.AP.yaw_damper_p_wo,
                        d1=1,
                        Ts=ts_control,
                        limit=self.AP.max_rudder)

        # instantiate TECS controllers
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
        self.Ts = ts_control
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        ###### TODO ######
        delta = MsgDelta(elevator=0,
                         aileron=0,
                         rudder=0,
                         throttle=0)
        
        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.saturate(  # why saturate when the controller will?
            cmd.phi_feedforward + self.course_from_roll.update(chi_c, state.chi),
            -self.AP.max_roll,
            self.AP.max_roll
        )
        delta.aileron = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta.rudder = self.yaw_damper.update(state.r)

        # longitudinal TECS autopilot
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
            delta.throttle = 0.
        elif Tc > self.thrust_max:
            delta.throttle = 1.
        else:
            throttle = self.throttle_lut((Tc, state.Va))
            if obj_fn(throttle, state.Va, Tc) > MAX_LUT_ERROR_SQ:
                # lookup has unacceptable error, use previous value
                delta.throttle = self.delta_t_d1
            else:
                delta.throttle = throttle

        # deal with steady-state error
        delta.throttle += self.throttle_correction_from_airspeed.update(cmd.airspeed_command, state.Va)

        # saturate
        delta.throttle = self.saturate(delta.throttle, 0, 1)

        ## pitch angle
        sin_gamma_c = h_des_dot / state.Va + ( (self.k_T - self.k_D) * E_K_tilde + (self.k_T + self.k_D) * E_P_tilde ) / ( 2 * MAV.mass * MAV.gravity * state.Va ) 
            
        sin_gamma_c_sat = self.saturate(sin_gamma_c, -np.sin(self.AP.max_pitch), np.sin(self.AP.max_pitch))
        # theta_c - alpha = gamma_a
        # - gamma_a is the air mass referenced flight path angle.  When wind velocity
        # - is 0, gamma = gamma_a.
        theta_c = state.alpha + np.arcsin(sin_gamma_c_sat)
        
        # deal with steady-state error
        theta_c += self.pitch_correction_from_altitude.update(cmd.altitude_command, state.altitude)
        delta.elevator = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        
        # construct output and commanded states
        self.commanded_state.altitude = cmd.altitude_command 
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        
        self.delta_t_d1 = delta.throttle
        
        return delta, self.commanded_state

    def saturate(self, _input, low_limit, up_limit):
        if _input <= low_limit:
            output = low_limit
        elif _input >= up_limit:
            output = up_limit
        else:
            output = _input
        return output
