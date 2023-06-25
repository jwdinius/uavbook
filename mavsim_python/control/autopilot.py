"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
# from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from control.pi_control import PIControl
from control.pd_control_with_rate import PDControlWithRate
from control.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral-directional controllers
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

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = PIControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = MsgState()
        self._trim_input = MsgDelta()

    def set_trim_input(self, trim_input):
        self._trim_input = trim_input

    def update(self, cmd, state):
        #### TODO #####
        # construct control outputs and commanded states
        #delta = MsgDelta(elevator=0,
        #                 aileron=0,
        #                 rudder=0,
        #                 throttle=0)
        delta = self._trim_input

        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.saturate(  # why saturate when the controller will?
            cmd.phi_feedforward + self.course_from_roll.update(chi_c, state.chi),
            -np.radians(45),
            np.radians(45)
        )
        delta.aileron = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta.rudder = self.yaw_damper.update(state.r)
        
        # longitudinal autopilot
        h_c = self.saturate(
            cmd.altitude_command,
            state.altitude - AP.altitude_zone,
            state.altitude + AP.altitude_zone
        )
        theta_c = self.altitude_from_pitch.update(h_c, state.altitude)
        delta.elevator = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        delta.throttle = self.saturate(
            self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va),
            0,
            1
        )

        self.commanded_state.altitude = cmd.altitude_command 
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command

        return delta, self.commanded_state

    def saturate(self, _input, low_limit, up_limit):
        if _input <= low_limit:
            output = low_limit
        elif _input >= up_limit:
            output = up_limit
        else:
            output = _input
        return output
