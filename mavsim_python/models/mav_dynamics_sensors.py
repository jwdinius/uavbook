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
from message_types.msg_sensors import MsgSensors
from message_types.msg_delta import MsgDelta

import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation
from scipy.optimize import minimize

class MavDynamics:
    def __init__(self, Ts, use_biases=False, debug=False):
        self._ts_simulation = Ts
        self._debug = debug
        self._use_biases = use_biases
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
        # initialize true_state message
        self.true_state = MsgState()
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.
        # reseed the random number generator
        np.random.seed(11011)
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
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

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
        gyro_x_bias = 0.
        gyro_y_bias = 0.
        gyro_z_bias = 0.
        mag_x_bias = 0.
        mag_y_bias = 0.
        mag_z_bias = 0.
        abs_pres_bias = 0.
        diff_pres_bias = 0.

        if self._use_biases:
            gyro_x_bias = SENSOR.gyro_x_bias
            gyro_y_bias = SENSOR.gyro_y_bias
            gyro_z_bias = SENSOR.gyro_z_bias
            mag_x_bias = SENSOR.mag_x_bias
            mag_y_bias = SENSOR.mag_y_bias
            mag_z_bias = SENSOR.mag_z_bias
            abs_pres_bias = SENSOR.abs_pres_bias
            diff_pres_bias = SENSOR.diff_pres_bias

        # simulate rate gyros(units are rad / sec)
        p, q, r = [self._state.item(i) for i in range(10, 13)]
        self._sensors.gyro_x = p + np.random.normal(loc=gyro_x_bias, scale=SENSOR.gyro_sigma)
        self._sensors.gyro_y = q + np.random.normal(loc=gyro_y_bias, scale=SENSOR.gyro_sigma)
        self._sensors.gyro_z = r + np.random.normal(loc=gyro_z_bias, scale=SENSOR.gyro_sigma)

        # simulate accelerometers(units of g)
        ax, ay, az = [self._forces.item(i) / MAV.mass for i in range(3)]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self._sensors.accel_x = ax + MAV.gravity * np.sin(theta) + np.random.normal(scale=SENSOR.accel_sigma) 
        self._sensors.accel_y = ay - MAV.gravity * np.cos(theta) * np.sin(phi) + np.random.normal(scale=SENSOR.accel_sigma) 
        self._sensors.accel_z = az - MAV.gravity * np.cos(theta) * np.cos(phi) + np.random.normal(scale=SENSOR.accel_sigma) 

        # simulate magnetometers
        # get magnetic north in the inertial frame
        m_i = SENSOR.R_m2i @ np.array([[1.], [0.], [0.]])
        R_b2i = Quaternion2Rotation(self._state[6:10])
        # get magnetic north in the body frame
        m_body = R_b2i.T @ m_i
        m_x, m_y, m_z = m_body.reshape((3,))
        # apply bias and noise
        self._sensors.mag_x = m_x + np.random.normal(loc=mag_x_bias, scale=SENSOR.mag_sigma)
        self._sensors.mag_y = m_y + np.random.normal(loc=mag_y_bias, scale=SENSOR.mag_sigma)
        self._sensors.mag_z = m_z + np.random.normal(loc=mag_z_bias, scale=SENSOR.mag_sigma)
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        
        if self._debug:
            m_b_hat = np.array([[self._sensors.mag_x], [self._sensors.mag_y], [self._sensors.mag_z]])
            # R_v2^v1 = (R_v1^v2).T from pg. 15 of book
            Rth = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]]) 
            # R_b^v2 = (R_v2^b).T from pg. 15 of book
            Rph = np.array([[1., 0., 0.], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]]) 
            m_v1_hat = Rth @ Rph @ m_b_hat
            def fn(x, m_i, b_v1):
                _psi = x[0]
                # R_i^v1
                Rps = np.array([[np.cos(_psi), np.sin(_psi), 0.], [-np.sin(_psi), np.cos(_psi), 0.], [0., 0., 1.]])
                m_v1 = Rps @ m_i
                return (m_v1[0, 0] - b_v1[0, 0])**2 + (m_v1[1, 0] - b_v1[1, 0])**2
            heading_soln = minimize(fn, psi, args=(m_i, m_v1_hat))
            print(f"+++ psi(true): {np.degrees(psi)}")
            if not heading_soln.success:
                print("heading root find failed")
            else:
                heading = np.degrees(heading_soln.x[0])
                print(f"### heading (alt): {heading}")
 

        # simulate pressure sensors
        h_AGL = self.true_state.altitude
        self._sensors.abs_pressure = MAV.rho * MAV.gravity * h_AGL + np.random.normal(loc=abs_pres_bias, scale=SENSOR.abs_pres_sigma)
        Va = self.true_state.Va
        self._sensors.diff_pressure = 0.5 * MAV.rho * Va**2 + np.random.normal(loc=diff_pres_bias, scale=SENSOR.diff_pres_sigma)
        
        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            last_gps_eta_n = self._gps_eta_n
            last_gps_eta_e = self._gps_eta_e
            last_gps_eta_h = self._gps_eta_h
            self._gps_eta_n = np.exp(-SENSOR.gps_k * self._ts_simulation) * last_gps_eta_n + np.random.normal(scale=SENSOR.gps_n_sigma)
            self._gps_eta_e = np.exp(-SENSOR.gps_k * self._ts_simulation) * last_gps_eta_e + np.random.normal(scale=SENSOR.gps_e_sigma)
            self._gps_eta_h = np.exp(-SENSOR.gps_k * self._ts_simulation) * last_gps_eta_h + np.random.normal(scale=SENSOR.gps_h_sigma)
            self._sensors.gps_n = self.true_state.north + self._gps_eta_n
            self._sensors.gps_e = self.true_state.east + self._gps_eta_e
            self._sensors.gps_h = self.true_state.altitude + self._gps_eta_h
            Vn = self.true_state.Vg * np.cos(self.true_state.chi) * np.cos(self.true_state.gamma)
            Ve = self.true_state.Vg * np.sin(self.true_state.chi) * np.cos(self.true_state.gamma)
            sigma_Vg = SENSOR.gps_Vg_sigma
            sigma_chi = sigma_Vg / np.sqrt(Vn**2 + Ve**2)
            sigma_gamma = sigma_Vg / self.true_state.Vg
            self._sensors.gps_Vg = self.true_state.Vg + np.random.normal(scale=sigma_Vg)
            self._sensors.gps_course = self.true_state.chi + np.random.normal(scale=sigma_chi)
            self._sensors.gps_gamma = self.true_state.gamma + np.random.normal(scale=sigma_gamma)
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
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
        steady_state = wind[0:3]
        gust = wind[3:6]
        
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

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ###### 
        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg_x, fg_y, fg_z = self._calc_gravity_forces()
        
        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        # compute longitudinal forces and moments in the body frame
        fa_x, fa_z, my = self._calc_longitudinal_forces_and_moments(delta)

        # compute lateral forces and moments in the body frame
        fa_y, mx, mz = self._calc_lateral_forces_and_moments(delta)

        Fx = fg_x + fa_x + thrust_prop
        Fy = fg_y + fa_y
        Fz = fg_z + fa_z
        Mx = mx + torque_prop
        My = my
        Mz = mz

        self._forces = np.array([[Fx], [Fy], [Fz]])
        forces_moments = np.array([[Fx, Fy, Fz, Mx, My, Mz]]).T
        return forces_moments
    
    def _calc_gravity_forces(self):
        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg = Quaternion2Rotation(self._state[6:10]).T @ np.array([[0], [0], [MAV.mass * MAV.gravity]])
        return [fg.item(i) for i in range(0, 3)]

    def _motor_thrust_torque(self, airspeed, delta_t):
        # compute thrust and torque due to propeller
        ##### TODO #####
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
        C_Q = MAV.C_Q2 * J_op**2 + MAV.C_Q1 * J_op + MAV.C_Q0

        n = Omega_op / (2 * np.pi)

        thrust_prop = MAV.rho * n**2 * MAV.D_prop**4 * C_T
        torque_prop = -MAV.rho * n**2 * MAV.D_prop**5 * C_Q  # XXX adding minus sign like in the book has the wrong sign compared to chap4_check output (?)

        return thrust_prop, torque_prop
    
    def _calc_longitudinal_forces_and_moments(self, delta):
        _, q, _ = [self._state.item(i) for i in range(10, 13)]

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
        F_drag = aero_force_scaling * (C_D + 0.5 * MAV.C_D_q * MAV.c / self._Va * q + MAV.C_D_delta_e * delta.elevator)
        self._F_drag = F_drag  # added for quick POC of tecs AP

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
        self.true_state.gamma = np.arcsin(-pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.wd = self._wind.item(2)
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        self.true_state.camera_az = self._state.item(13)
        self.true_state.camera_el = self._state.item(14)
        # to get an idea of model error with model on pg. 157:
        #with open("accel_errors.txt", "a") as f:
        #    p, q, r = [self._state.item(i) for i in range(10, 13)]
        #    ax, ay, az = [self._forces.item(i) / MAV.mass for i in range(3)]
        #    Va = self._Va 
        #    err_x = ax - q * Va * np.sin(theta)
        #    err_y = ay - (r * Va * np.cos(theta) - p* Va * np.sin(theta))
        #    err_z = az - (-q * Va * np.cos(theta))
        #    f.write(f"{err_x}, {err_y}, {err_z}\n")
