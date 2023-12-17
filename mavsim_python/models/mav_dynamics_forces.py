"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np

# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation

class MavDynamics:
    def __init__(self, Ts, debug=False):
        self._ts_simulation = Ts
        self._debug = debug
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # initialize true_state message
        self.true_state = MsgState()


    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state[0:13], forces_moments)
        k2 = self._derivatives(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the message class for the true state
        self._update_true_state()


    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        ##### TODO #####
        # Extract the States
        n, e, d = [state.item(i) for i in range(0, 3)]
        u, v, w = [state.item(i) for i in range(3, 6)]
        e0, e1, e2, e3 = [state.item(i) for i in range(6, 10)]
        # normalize quaternion (not needed, update call will do the normalization)
        #e = np.array([e0, e1, e2, e3])
        #e0, e1, e2, e3 = e / np.linalg.norm(e)
        p, q, r = [state.item(i) for i in range(10, 13)]

        # Extract Forces/Moments
        # in reference to the book: l == mx, m == my, n == mz
        fx, fy, fz = [forces_moments.item(i) for i in range(0, 3)]
        mx, my, mz = [forces_moments.item(i) for i in range(3, 6)]

        # see Appendix B.2
        # Position Kinematics
        # compute ei * ej and put it into a 4x4 matrix
        e_e = np.array([[i * j for j in (e0, e1, e2, e3)] for i in (e0, e1, e2, e3)])
        #e_e = np.array([[e0*e0, e0*e1, e0*e2, e0*e3],
        #                [e1*e0, e1*e1, e1*e2, e1*e3],
        #                [e2*e0, e2*e1, e2*e2, e2*e3],
        #                [e3*e0, e3*e1, e3*e2, e3*e3]])

        n_dot = (e_e[1, 1] + e_e[0, 0] - e_e[2, 2] - e_e[3, 3]) * u \
            + 2 * (e_e[1, 2] - e_e[3, 0]) * v \
            + 2 * (e_e[1, 3] + e_e[2, 0]) * w
        e_dot = 2 * (e_e[1, 2] + e_e[3, 0]) * u \
            + (e_e[2, 2] + e_e[0, 0] - e_e[1, 1] - e_e[3, 3]) * v \
            + 2 * (e_e[2, 3] - e_e[1, 0]) * w
        d_dot = 2 * (e_e[1, 3] - e_e[2, 0]) * u \
            + 2 * (e_e[2, 3] + e_e[1, 0]) * v \
            + (e_e[3, 3] + e_e[0, 0] - e_e[1, 1] - e_e[2, 2]) * w

        # Position Dynamics
        u_dot = r * v - q * w + fx / MAV.mass
        v_dot = p * w - r * u + fy / MAV.mass
        w_dot = q * u - p * v + fz / MAV.mass

        # rotational kinematics
        e0_dot = -0.5 * (p * e1 + q * e2 + r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # rotatonal dynamics
        p_dot = MAV.gamma1 * p * q - MAV.gamma2 * q * r + MAV.gamma3 * mx + MAV.gamma4 * mz
        q_dot = MAV.gamma5 * p * r - MAV.gamma6 * (p * p - r * r) + my / MAV.Jy
        r_dot = MAV.gamma7 * p * q - MAV.gamma1 * q * r + MAV.gamma4 * mx + MAV.gamma8 * mz
        
        # collect the derivative of the states
        # x_dot = np.array([[north_dot, east_dot,... ]]).T
        x_dot = np.array([
            [n_dot,
             e_dot,
             d_dot,
             u_dot,
             v_dot,
             w_dot,
             e0_dot,
             e1_dot,
             e2_dot,
             e3_dot,
             p_dot,
             q_dot,
             r_dot]
        ]).T
        
        if self._debug:
            _x_dot = self._derivatives_reference(state, forces_moments)
            assert np.isclose(n_dot, _x_dot.item(0))
            assert np.isclose(e_dot, _x_dot.item(1))
            assert np.isclose(d_dot, _x_dot.item(2))
            assert np.isclose(u_dot, _x_dot.item(3))
            assert np.isclose(v_dot, _x_dot.item(4))
            assert np.isclose(w_dot, _x_dot.item(5))
            assert np.isclose(e0_dot, _x_dot.item(6))
            assert np.isclose(e1_dot, _x_dot.item(7))
            assert np.isclose(e2_dot, _x_dot.item(8))
            assert np.isclose(e3_dot, _x_dot.item(9))
            assert np.isclose(p_dot, _x_dot.item(10))
            assert np.isclose(q_dot, _x_dot.item(11))
            assert np.isclose(r_dot, _x_dot.item(12))

        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0

    def _derivatives_reference(self, state, forces_moments):
        """
        XXX solution from https://github.com/b4sgren/MAV_Autopilot/blob/498268208fd3dbd1fa69d1fc581b4a4d66d9734c/chp3/mav_dynamics.py
        pulled to compare my solution with

        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        Rv_b = np.array([[e1**2 + e0**2 - e2**2 - e3**2, 2*(e1*e2 - e3*e0), 2*(e1*e3 + e2*e0)],
                         [2*(e1*e2 + e3*e0), e2**2 + e0**2 - e1**2 - e3**2, 2*(e2*e3 - e1*e0)],
                         [2*(e1*e3 - e2*e0), 2*(e2*e3 + e1*e0), e3**2 + e0**2 - e1**2 - e2**2]])

        pos_dot = Rv_b @ np.array([u, v, w]).T
        pn_dot = pos_dot.item(0)
        pe_dot = pos_dot.item(1)
        pd_dot = pos_dot.item(2)

        # position dynamics
        u_dot = r*v - q*w + 1/MAV.mass * fx
        v_dot = p*w - r*u + 1/MAV.mass * fy
        w_dot = q*u - p*v + 1/MAV.mass * fz

        # rotational kinematics
        e0_dot = (-p * e1 - q * e2 - r * e3) * 0.5
        e1_dot = (p * e0 + r * e2 - q * e3) * 0.5
        e2_dot = (q * e0 - r * e1 + p * e3) * 0.5
        e3_dot = (r * e0 + q * e1 - p * e2) * 0.5

        # rotatonal dynamics
        p_dot = MAV.gamma1 * p * q - MAV.gamma2 * q * r + MAV.gamma3 * l + MAV.gamma4 * n
        q_dot = MAV.gamma5 * p * r - MAV.gamma6 * (p**2 - r**2) + 1/MAV.Jy * m
        r_dot = MAV.gamma7 * p * q - MAV.gamma1 * q * r + MAV.gamma4 * l + MAV.gamma8 * n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot
