"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation

#XXX
from math import asin, exp

class MavDynamics:
    def __init__(self, Ts, use_lo_fi_thrust_model=False, debug=False):
        self._debug = debug
        self._ts_simulation = Ts
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
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize other before function calls
        self._use_lo_fi_thrust_model = use_lo_fi_thrust_model
        # initialize true_state message
        self.true_state = MsgState()
        # update velocity data 
        self._update_velocity_data()
        # update true state before updating forces and moments
        self._update_true_state()
        # update forces and moments 
        self._forces_moments(delta=MsgDelta())


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta.aileron, delta.elevator, delta.rudder, delta.throttle) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

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

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

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
        #  -- Copy From mav_dynamic_forces.py --
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
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]  # in "vehicle" (really, NED)
        gust = wind[3:6]  # in body

        ##### TODO #####
        # convert wind vector from world to body frame (self._wind = ?)
        R_b2v = Quaternion2Rotation(self._state[6:10])
        self._wind = steady_state + R_b2v @ gust 

        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        wind_b = R_b2v.T @ self._wind
        airspeed_vector = self._state[3:6] - wind_b
        ur, vr, wr = airspeed_vector

        # compute airspeed (self._Va = ?)
        self._Va = np.linalg.norm(airspeed_vector) 

        # compute angle of attack (self._alpha = ?)
        self._alpha = np.arctan2(float(wr), float(ur))
        
        # compute sideslip angle (self._beta = ?)
        self._beta = np.arcsin(float(vr) / self._Va)

    def _calc_gravity_forces(self):
        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg = Quaternion2Rotation(self._state[6:10]).T @ np.array([[0], [0], [MAV.mass * MAV.gravity]])
        return [fg.item(i) for i in range(0, 3)]

    def _calc_longitudinal_forces_and_moments(self, delta):
        p, q, r = [self._state.item(i) for i in range(10, 13)]
        M = MAV.M
        alpha = self._alpha
        alpha0 = MAV.alpha0
        rho = MAV.rho
        Va = self._Va
        S = MAV.S_wing
        #_q = self.true_state.q
        _q = q
        c = MAV.c
        de = delta.elevator

        sigma_alpha = (1 + exp(-M * (alpha - alpha0)) + exp(M * (alpha + alpha0))) /\
                      ((1 + exp(-M * (alpha - alpha0)))*(1 + exp(M * (alpha + alpha0))))
        CL_alpha = (1 - sigma_alpha) * (MAV.C_L_0 + MAV.C_L_alpha * alpha) + \
                    sigma_alpha * (2 * np.sign(alpha) * (np.sin(alpha)**2) * np.cos(alpha))

        _F_lift = 0.5 * rho * (Va**2) * S * (CL_alpha + MAV.C_L_q * (c / (2 * Va)) * _q \
                 + MAV.C_L_delta_e * de)
        CD_alpha = MAV.C_D_p + ((MAV.C_L_0 + MAV.C_L_alpha * alpha)**2) / (np.pi * MAV.e * MAV.AR)

        _F_drag = 0.5 * rho * (Va**2) * S * (CD_alpha + MAV.C_D_q * (c / (2 * Va)) * q \
                 + MAV.C_D_delta_e * de)
        #p, q, r = [self._state.item(i) for i in range(10, 13)]
        # compute Lift and Drag coefficients (CL, CD) (pgs. 47-48)
        ## Lift coefficient: C_L(\alpha)
        def sigma(alpha):
            num = 1. + np.exp(-MAV.M*(alpha-MAV.alpha0)) + np.exp(MAV.M*(alpha+MAV.alpha0))
            den = (1. + np.exp(-MAV.M*(alpha-MAV.alpha0))) * (1. + np.exp(MAV.M*(alpha+MAV.alpha0)))
            return num / den

        #C_L_alpha = np.pi * MAV.AR / (1. + np.sqrt(1 + (0.5 * MAV.AR)**2))
        C_L = (1. - sigma(self._alpha)) * (MAV.C_L_0 + MAV.C_L_alpha * self._alpha) \
            + 2. * sigma(self._alpha) * np.sign(self._alpha) * np.sin(self._alpha)**2 * np.cos(self._alpha)
        ## Drag coefficient: C_D(\alpha)
        C_D = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * self._alpha)**2 / (np.pi * MAV.e * MAV.AR)

        # compute Lift and Drag Forces (F_lift, F_drag)
        aero_force_scaling = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing
        F_lift = aero_force_scaling * (C_L + 0.5 * MAV.C_L_q * MAV.c / self._Va * q + MAV.C_L_delta_e * delta.elevator) 
        assert np.isclose(_F_lift, F_lift)
        F_drag = aero_force_scaling * (C_D + 0.5 * MAV.C_D_q * MAV.c / self._Va * q + MAV.C_D_delta_e * delta.elevator)
        assert np.isclose(_F_drag, F_drag)

        # forces, moments = sum(gravity, aero, propeller)
        ca = np.cos(self._alpha)
        sa = np.sin(self._alpha)
        # compute longitudinal forces in body frame (fx, fz)
        fx = -F_drag * ca + F_lift * sa
        fz = -F_drag * sa - F_lift * ca 
        
        # compute logitudinal torque in body frame (My) ( == "m" in the book)
        my = aero_force_scaling * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha * self._alpha + 0.5 * MAV.C_m_q * MAV.c / self._Va * q + MAV.C_m_delta_e * delta.elevator)
        return fx, fz, my

    def _calc_lateral_forces_and_moments(self, delta):
        p, q, r = [self._state.item(i) for i in range(10, 13)]
        aero_force_scaling = 0.5 * MAV.rho * self._Va**2 * MAV.S_wing
        fy = aero_force_scaling * (MAV.C_Y_0 + MAV.C_Y_beta * self._beta + 0.5 * MAV.C_Y_p * MAV.b * p / self._Va + 0.5 * MAV.C_Y_r * MAV.b * r / self._Va + MAV.C_Y_delta_a * delta.aileron + MAV.C_Y_delta_r * delta.rudder)

        # compute lateral torques in body frame (Mx, Mz) ( == "(l, n) in the book)")
        mx = aero_force_scaling * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta * self._beta + 0.5 * MAV.C_ell_p * MAV.b * p / self._Va + 0.5 * MAV.C_ell_r * MAV.b * r / self._Va + MAV.C_ell_delta_a * delta.aileron + MAV.C_ell_delta_r * delta.rudder)
        mz = aero_force_scaling * MAV.b * (MAV.C_n_0 + MAV.C_n_beta * self._beta + 0.5 * MAV.C_n_p * MAV.b * p / self._Va + 0.5 * MAV.C_n_r * MAV.b * r / self._Va + MAV.C_n_delta_a * delta.aileron + MAV.C_n_delta_r * delta.rudder)
        return fy, mx, mz

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta.aileron, delta.elevator, delta.rudder, delta.throttle)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)
        phi, theta, psi = Quaternion2Euler(self._state[6:10])

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg_x, fg_y, fg_z = self._calc_gravity_forces()
        
        if self._debug:
            fb_grav = Quaternion2Rotation(self._state[6:10]).T @ np.array([[0, 0, MAV.mass * MAV.gravity]]).T
            assert np.isclose(fb_grav[0], fg_x)
            assert np.isclose(fb_grav[1], fg_y)
            assert np.isclose(fb_grav[2], fg_z)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(delta.throttle)
        if self._debug:
            Fp, Qp = self.calcThrustForceAndMoment(delta.throttle)
            assert np.isclose(Fp, thrust_prop)
            assert np.isclose(Qp, torque_prop)

        # compute longitudinal forces and moments in the body frame
        fa_x, fa_z, my = self._calc_longitudinal_forces_and_moments(delta)
        if self._debug:
            _fa_x, _fa_z, _my = self.calcLongitudinalForcesAndMoments(delta.elevator)
            assert np.isclose(fa_x, _fa_x)
            assert np.isclose(fa_z, _fa_z)
            assert np.isclose(my, _my)

        # compute lateral forces and moments in the body frame
        fa_y, mx, mz = self._calc_lateral_forces_and_moments(delta)
        if self._debug:
            _fa_y, _mx, _mz = self.calcLateralForcesAndMoments(delta.aileron, delta.rudder)
            assert np.isclose(fa_y, _fa_y)
            assert np.isclose(mx, _mx)
            assert np.isclose(mz, _mz)

        Fx = fg_x + fa_x + thrust_prop
        Fy = fg_y + fa_y
        Fz = fg_z + fa_z
        Mx = mx + torque_prop
        My = my
        Mz = mz

        forces_moments = np.array([[Fx, Fy, Fz, Mx, My, Mz]]).T
        return forces_moments

    def _motor_thrust_torque(self, delta_t):
        # compute thrust and torque due to propeller
        ##### TODO #####
        if not self._use_lo_fi_thrust_model:
            # XXX For high fidelity model details; see Chapter 4 slides here https://drive.google.com/file/d/1BjJuj8QLWV9E1FX6sHVHXIGaIizUaAJ5/view?usp=sharing (particularly Slides 30-35)
            # map delta.throttle throttle command(0 to 1) into motor input voltage
            #v_in =
            V_in = MAV.V_max * delta_t

            # Angular speed of propeller (omega_p = ?)
            a = MAV.C_Q0 * MAV.rho * MAV.D_prop**5 / (2 * np.pi)**2
            b = MAV.rho * MAV.D_prop**4 * MAV.C_Q1 * self._Va / (2 * np.pi)  + (MAV.KQ * MAV.KV) / MAV.R_motor
            c = MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * self._Va**2 - (MAV.KQ * V_in) / MAV.R_motor + MAV.KQ * MAV.i0

            # use the positive root
            Omega_op = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

            # thrust and torque due to propeller
            J_op = (2 * np.pi * self._Va) / (Omega_op * MAV.D_prop)

            C_T = MAV.C_T2 * J_op**2 + MAV.C_T1 * J_op + MAV.C_T0
            C_Q = MAV.C_Q2 * J_op**2 + MAV.C_Q1 * J_op + MAV.C_Q0

            n = Omega_op /(2 * np.pi)

            thrust_prop = MAV.rho * n**2 * MAV.D_prop**4 * C_T
            torque_prop = MAV.rho * n**2 * MAV.D_prop**5 * C_Q  # XXX adding minus sign like in the book has the wrong sign compared to chap4_check output (?)
        else:
            thrust_prop = 0.5 * MAV.rho * MAV.S_prop * MAV.C_prop * ( (MAV.k_motor * delta_t)**2 - self._Va**2 ) 
            torque_prop = MAV.k_T_p * (MAV.k_omega * delta.throttle)**2

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0

    def calcThrustForceAndMoment(self, dt):
        '''from https://github.com/b4sgren/MAV_Autopilot/blob/498268208fd3dbd1fa69d1fc581b4a4d66d9734c/chp4/mav_dynamics.py
        included for debugging'''
        rho = MAV.rho
        D = MAV.D_prop
        Va = self._Va

        V_in = MAV.V_max * dt

        a = (rho * D**5) / ((2 * np.pi)**2) * MAV.C_Q0
        b = (rho * (D**4) * MAV.C_Q1 * Va)/(2 * np.pi)  + (MAV.KQ **2)/MAV.R_motor
        c = rho * (D**3) * MAV.C_Q2 * (Va**2) - (MAV.KQ * V_in)/MAV.R_motor + MAV.KQ * MAV.i0

        Omega_op = (-b + np.sqrt((b**2) - 4 * a * c)) / (2 * a)
        J_op = (2 * np.pi * Va) / (Omega_op * D)

        CT = MAV.C_T2 * (J_op**2) + MAV.C_T1 * J_op + MAV.C_T0
        CQ = MAV.C_Q2 * (J_op**2) + MAV.C_Q1 * J_op + MAV.C_Q0

        Qp = rho * (Omega_op / (2 * np.pi))**2 * (D**5) * CQ
        Fp = CT * (rho * (Omega_op**2) * (D**4)) / ((2 * np.pi)**2)

        return Fp, Qp

    def calcLateralForcesAndMoments(self, da, dr):
        '''from https://github.com/b4sgren/MAV_Autopilot/blob/498268208fd3dbd1fa69d1fc581b4a4d66d9734c/chp4/mav_dynamics.py
        included for debugging'''
        b = MAV.b
        Va = self._Va
        #beta = self.true_state.beta
        #p = self.true_state.p
        #r = self.true_state.r
        beta = self._beta
        p = self._state.item(10)
        r = self._state.item(12)
        rho = MAV.rho
        S = MAV.S_wing

        # Calculating fy
        fy = 1/2.0 * rho * (Va**2) * S * (MAV.C_Y_0 + MAV.C_Y_beta * beta + MAV.C_Y_p * (b / (2*Va)) * p +\
             MAV.C_Y_r * (b / (2 * Va)) * r + MAV.C_Y_delta_a * da + MAV.C_Y_delta_r * dr)

        # Calculating l
        l = 1/2.0 * rho * (Va**2) * S * b * (MAV.C_ell_0 + MAV.C_ell_beta * beta + MAV.C_ell_p * (b/(2*Va)) * p +\
            MAV.C_ell_r * (b/(2*Va)) * r + MAV.C_ell_delta_a * da + MAV.C_ell_delta_r * dr)

        # Calculating n
        n = 1/2.0 * rho * (Va**2) * S * b * (MAV.C_n_0 + MAV.C_n_beta * beta + MAV.C_n_p * (b/(2*Va)) * p +\
            MAV.C_n_r * (b/(2*Va)) * r + MAV.C_n_delta_a * da + MAV.C_n_delta_r * dr)

        return fy, l, n

    def calcLongitudinalForcesAndMoments(self, de):
        '''from https://github.com/b4sgren/MAV_Autopilot/blob/498268208fd3dbd1fa69d1fc581b4a4d66d9734c/chp4/mav_dynamics.py
        included for debugging'''
        M = MAV.M
        alpha = self._alpha
        alpha0 = MAV.alpha0
        rho = MAV.rho
        Va = self._Va
        S = MAV.S_wing
        #q = self.true_state.q  # FIxME why isn't this updated with the _update_true_state() call in the constructor?
        q = self._state.item(11)
        c = MAV.c

        sigma_alpha = (1 + exp(-M * (alpha - alpha0)) + exp(M * (alpha + alpha0))) /\
                      ((1 + exp(-M * (alpha - alpha0)))*(1 + exp(M * (alpha + alpha0))))
        CL_alpha = (1 - sigma_alpha) * (MAV.C_L_0 + MAV.C_L_alpha * alpha) + \
                    sigma_alpha * (2 * np.sign(alpha) * (np.sin(alpha)**2) * np.cos(alpha))

        F_lift = 0.5 * rho * (Va**2) * S * (CL_alpha + MAV.C_L_q * (c / (2 * Va)) * q \
                 + MAV.C_L_delta_e * de)

        CD_alpha = MAV.C_D_p + ((MAV.C_L_0 + MAV.C_L_alpha * alpha)**2) / (np.pi * MAV.e * MAV.AR)

        F_drag = 0.5 * rho * (Va**2) * S * (CD_alpha + MAV.C_D_q * (c / (2 * Va)) * q \
                 + MAV.C_D_delta_e * de)

        Rb_s = np.array([[np.cos(alpha), -np.sin(alpha)],
                         [np.sin(alpha), np.cos(alpha)]])
        fx_fz = Rb_s @ np.array([[-F_drag, -F_lift]]).T

        m = 0.5 * rho * (Va**2) * S * c * (MAV.C_m_0 + MAV.C_m_alpha * alpha + \
            MAV.C_m_q * (c / (2 * Va)) * q + MAV.C_m_delta_e * de)

        return fx_fz.item(0), fx_fz.item(1), m
