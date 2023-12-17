"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
from dotenv import load_dotenv
load_dotenv()
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.append(os.environ["UAVBOOK_HOME"])
import numpy as np
from tools.wrap import wrap
from control.pi_control import PIControl
from control.pd_control_with_rate import PDControlWithRate
from control.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from importlib import import_module


class Autopilot:
    def __init__(self, ts_control, use_truth=False):
        if use_truth:
            self.AP = import_module("parameters.control_parameters")
        else:
            self.AP = import_module("parameters.control_parameters_estimator")
        # instantiate lateral-directional controllers
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

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=self.AP.pitch_kp,
                        kd=self.AP.pitch_kd,
                        limit=self.AP.max_elevator)
        self.altitude_from_pitch = PIControl(
                        kp=self.AP.altitude_kp,
                        ki=self.AP.altitude_ki,
                        Ts=ts_control,
                        limit=self.AP.max_pitch)
        self.airspeed_from_throttle = PIControl(
                        kp=self.AP.airspeed_throttle_kp,
                        ki=self.AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        #### TODO #####
        # construct control outputs and commanded states
        #delta = MsgDelta(elevator=0,
        #                 aileron=0,
        #                 rudder=0,
        #                 throttle=0)
        delta = MsgDelta()

        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.saturate(  # why saturate when the controller will?
            cmd.phi_feedforward + self.course_from_roll.update(chi_c, state.chi),
            -self.AP.max_roll,
            self.AP.max_roll
        )
        delta.aileron = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta.rudder = self.yaw_damper.update(state.r)
        
        # longitudinal autopilot
        h_c = self.saturate(
            cmd.altitude_command,
            state.altitude - self.AP.altitude_zone,
            state.altitude + self.AP.altitude_zone
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
