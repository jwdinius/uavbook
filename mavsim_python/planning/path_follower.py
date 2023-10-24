import numpy as np
from math import sin, cos
import sys

sys.path.append('..')
import parameters.aerosonde_parameters as MAV
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap
from tools.rotations import Euler2Rotation

class PathFollower:
    def __init__(self):
        ##### TODO #####
        self.chi_inf = np.radians(45)  # approach angle for large distance from straight-line path
        self.k_path = 0.15  # path gain for straight-line path following
        self.k_orbit = 1.5  # path gain for orbit following
        self.gravity = MAV.gravity
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        ##### TODO #####
        p = np.array([[state.north], [state.east], [-state.altitude]])
        q = path.line_direction
        r = path.line_origin
        e = (p - r).reshape((3,))
        ki = np.array([[0.], [0.], [1.]])

        #airspeed command
        self.autopilot_commands.airspeed_command = MAV.Va0

        # course command
        chi_q = wrap(np.arctan2(q.item(1), q.item(0)), state.chi)
        while chi_q - state.chi < -np.pi:
            chi_q += 2*np.pi
        while chi_q - state.chi > np.pi:
            chi_q -= 2*np.pi
        R_p2i = Euler2Rotation(0, 0, chi_q)
        ep = R_p2i.T @ e
        chi_c = float(chi_q - self.chi_inf * (2. / np.pi) * np.arctan(self.k_path * ep.item(1)))
        self.autopilot_commands.course_command = wrap(chi_c, state.chi)

        # altitude command
        n = np.cross(q.reshape((3,)), ki.reshape((3,)))
        n = n / np.linalg.norm(n)
        s = e - e.dot(n)*n
        self.autopilot_commands.altitude_command = float(-r[2] + np.sqrt(s[0]**2 + s[1]**2) * ( q.item(2) / np.sqrt(q.item(0)**2 + q.item(1)**2)))

        # feedforward roll angle for straight line is zero
        self.autopilot_commands.phi_feedforward = 0

    def _follow_orbit(self, path, state):
        ##### TODO #####
        p = np.array([[state.north], [state.east], [-state.altitude]])
        c = path.orbit_center
        rho = path.orbit_radius
        lam = -1 if path.orbit_direction == "CW" else 1
        e = (p - c).reshape((3,))
        d = np.sqrt(e[0]**2 + e[1]**2)

        varpsi = wrap(np.arctan2(e[1], e[0]), state.chi)
        while varpsi - state.chi < -np.pi:
            varpsi += 2*np.pi
        while varpsi - state.chi > np.pi:
            varpsi -= 2*np.pi
        
        # airspeed command
        self.autopilot_commands.airspeed_command = MAV.Va0

        # course command
        chi_c = float(varpsi + lam * (0.5 * np.pi + np.arctan(self.k_orbit * ( (d - rho) / rho ))))
        self.autopilot_commands.course_command = wrap(chi_c, state.chi)

        # altitude command
        self.autopilot_commands.altitude_command = -float(c.item(2))
        
        # roll feedforward command
        self.autopilot_commands.phi_feedforward = 0




