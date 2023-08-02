"""
autopilot block for mavsim_python - Total Energy Control System
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/14/2020 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV
from control.tf_control import TFControl
from tools.wrap import wrap
from control.pi_control import PIControl
from control.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from scipy.optimize import root

def saturate(inp, low_limit, up_limit):
    if inp <= low_limit:
        output = low_limit
    elif inp >= up_limit:
        output = up_limit
    else:
        output = inp
    return output

def calculate_motor_thrust(delta_t, airspeed):
    # compute thrust due to propeller (copied from mav_dynamics_control)
    # XXX For high fidelity model details; see Chapter 4 slides here https://drive.google.com/file/d/1BjJuj8QLWV9E1FX6sHVHXIGaIizUaAJ5/view?usp=sharing (particularly Slides 30-35)
    # map delta.throttle throttle command(0 to 1) into motor input voltage
    #v_in =
    V_in = MAV.V_max * delta_t

    # Angular speed of propeller (omega_p = ?)
    a = MAV.C_Q0 * MAV.rho * MAV.D_prop**5 / (2 * np.pi)**2
    b = MAV.rho * MAV.D_prop**4 * MAV.C_Q1 * airspeed / (2 * np.pi)  + MAV.KQ**2 / MAV.R_motor
    c = MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * airspeed**2 - (MAV.KQ * V_in) / MAV.R_motor + MAV.KQ * MAV.i0

    # use the positive root
    Omega_op = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    # thrust and torque due to propeller
    J_op = (2 * np.pi * airspeed) / (Omega_op * MAV.D_prop)

    C_T = MAV.C_T2 * J_op**2 + MAV.C_T1 * J_op + MAV.C_T0

    n = Omega_op / (2 * np.pi)

    return MAV.rho * n**2 * MAV.D_prop**4 * C_T  # == thrust_prop

def compute_throttle_from_thrust(T_des, x_init, airspeed):
    def fn(_delta_t, _airspeed, _T_des):
        return calculate_motor_thrust(_delta_t, _airspeed) - _T_des
    optimal_delta_t = root(fn, x_init, args=(airspeed, T_des))
    if not optimal_delta_t.success:
        return None
    return optimal_delta_t.x[0]

class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        # self.yaw_damper = TransferFunction(
        #                 num=np.array([[AP.yaw_damper_kr, 0]]),
        #                 den=np.array([[1, AP.yaw_damper_p_wo]]),
        #                 Ts=ts_control)
        self.yaw_damper = TFControl(
                        k=AP.yaw_damper_kr,
                        n0=0.0,
                        n1=1.0,
                        d0=AP.yaw_damper_p_wo,
                        d1=1,
                        Ts=ts_control)

        # instantiate TECS controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp_tecs,
                        kd=AP.pitch_kd_tecs,
                        limit=np.radians(45))
        self.gamma_lyapunov = 1.0  # how hard to drive the Lyapunov function to 0: V_dot = -gamma * V, which has solution V(0)*exp(-gamma*t)
        self.k_Va = 0.1  # 1 / time constant for first-order lag on desired airspeed error dynmamics
        self.k_h = 0.1  # 1 / time constant for first-order lag on desired altitude error dynamics
        self.delta_t_d1 = 0.5
        self.theta_c_max = np.radians(45)
        self.Ts = ts_control
        self.commanded_state = MsgState()

    def set_trim_input(self, trim_input):
        self._trim_input = trim_input

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
            -np.radians(45),
            np.radians(45)
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
        print(f"-- airspeed_error = {airspeed_error}")
        print(f"-- altitude_error = {altitude_error}")
        Va_des_dot = state.Va_dot - self.k_Va * airspeed_error
        h_des_dot = state.h_dot - self.k_h * altitude_error

        E_K_des_dot = MAV.mass * cmd.airspeed_command * Va_des_dot 
        E_P_des_dot = MAV.mass * MAV.gravity * h_des_dot
        E_T_des_dot = E_K_des_dot + E_P_des_dot
        E_K_tilde = E_K_des - E_K
        E_P_tilde = E_P_des - E_P
        E_T_tilde = E_K_tilde + E_P_tilde  # total energy error
        
        ## thrust
        #T_drag = -state.F_drag  # equal and opposite thrust to oppose drag
        print(f"-- T_drag = {state.F_drag}")
        print(f"-- E_T_tilde = {E_T_tilde}")
        print(f"-- E_T_des_dot = {E_T_des_dot}")
        Tc = state.F_drag + (E_T_des_dot + self.gamma_lyapunov * E_T_tilde) / state.Va
        #Tc = calculate_motor_thrust(self._trim_input.throttle, state.Va)
        print(f"-- Tc = {Tc}")
        # convert thrust command to throttle command
        ## solve T_c - calculate_motor_thrust(state.Va, delta_t) = 0 for delta_t
        # initialize the root finder with previous estimate
        delta_t = compute_throttle_from_thrust(Tc, self.delta_t_d1, state.Va)
        if delta_t is None:
            print("Throttle calculation FAILED")
        else:
            print(f"-- Unsaturated throttle = {delta_t}")
            delta.throttle = saturate(delta_t, 0, 1)
            print(f"-- Saturated throttle {delta.throttle}")
            self.delta_t_d1 = delta.throttle

        ## pitch angle
        print(f"-- E_P_des_dot = {E_P_des_dot}")
        print(f"-- gamma*E_P_tilde = {self.gamma_lyapunov*E_P_tilde}")
        sin_theta_c = (E_P_des_dot + self.gamma_lyapunov * E_P_tilde) / (MAV.mass * MAV.gravity * state.Va)
        #print(f"+++++ {sin_theta_c}")
        if abs(sin_theta_c) > 1.:
            print("sin_theta_c is too large")

        sin_theta_c_sat = saturate(sin_theta_c, -np.sin(self.theta_c_max), np.sin(self.theta_c_max))
        theta_c = np.arcsin(sin_theta_c_sat)
        #theta_c = state.theta
        print(f"-- theta_c = {theta_c}")
        delta.elevator = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        #delta.elevator = self._trim_input.elevator
        print(f"-- elevator = {delta.elevator}")
        # construct output and commanded states
        self.commanded_state.altitude = cmd.altitude_command 
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
