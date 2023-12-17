"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    A_lon_w_alpha, B_lon_w_alpha, A_lat_w_beta, B_lat_w_beta = compute_ss_model(mav, trim_state, trim_input, with_beta=True, with_alpha=True) 
    # compute eigenvalues of A_lat and A_lon
    lon_eigvals, _ = np.linalg.eig(A_lon)
    icsi_lon0, wn_lon0 = convert_complex_conj_pair(lon_eigvals[1], lon_eigvals[2])
    icsi_lon1, wn_lon1 = convert_complex_conj_pair(lon_eigvals[3], lon_eigvals[4])
    lat_eigvals, _ = np.linalg.eig(A_lat)
    icsi_lat, wn_lat = convert_complex_conj_pair(lat_eigvals[3], lat_eigvals[4])
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('#A_lon_eigvals = np.array([%f + %f * j, %f + %f * j, %f + %f * j, %f + %f * j, %f + %f * j])\n' % \
               (np.real(lon_eigvals[0]), np.imag(lon_eigvals[0]), np.real(lon_eigvals[1]), np.imag(lon_eigvals[1]), \
                np.real(lon_eigvals[2]), np.imag(lon_eigvals[2]), np.real(lon_eigvals[3]), np.imag(lon_eigvals[3]), \
                np.real(lon_eigvals[4]), np.imag(lon_eigvals[4])))
    file.write(f'#A_lon_mode_1: \icsi = {icsi_lon0}, \w_n = {wn_lon0}\n')
    file.write(f'#A_lon_mode_2: \icsi = {icsi_lon1}, \w_n = {wn_lon1}\n')
    file.write('A_lon_w_alpha = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon_w_alpha[0][0], A_lon_w_alpha[0][1], A_lon_w_alpha[0][2], A_lon_w_alpha[0][3], A_lon_w_alpha[0][4],
     A_lon_w_alpha[1][0], A_lon_w_alpha[1][1], A_lon_w_alpha[1][2], A_lon_w_alpha[1][3], A_lon_w_alpha[1][4],
     A_lon_w_alpha[2][0], A_lon_w_alpha[2][1], A_lon_w_alpha[2][2], A_lon_w_alpha[2][3], A_lon_w_alpha[2][4],
     A_lon_w_alpha[3][0], A_lon_w_alpha[3][1], A_lon_w_alpha[3][2], A_lon_w_alpha[3][3], A_lon_w_alpha[3][4],
     A_lon_w_alpha[4][0], A_lon_w_alpha[4][1], A_lon_w_alpha[4][2], A_lon_w_alpha[4][3], A_lon_w_alpha[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('B_lon_w_alpha = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon_w_alpha[0][0], B_lon_w_alpha[0][1],
     B_lon_w_alpha[1][0], B_lon_w_alpha[1][1],
     B_lon_w_alpha[2][0], B_lon_w_alpha[2][1],
     B_lon_w_alpha[3][0], B_lon_w_alpha[3][1],
     B_lon_w_alpha[4][0], B_lon_w_alpha[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('#A_lat_eigvals = np.array([%f + %f * j, %f + %f * j, %f + %f * j, %f + %f * j, %f + %f * j])\n' % \
               (np.real(lat_eigvals[0]), np.imag(lat_eigvals[0]), np.real(lat_eigvals[1]), np.imag(lat_eigvals[1]), \
                np.real(lat_eigvals[2]), np.imag(lat_eigvals[2]), np.real(lat_eigvals[3]), np.imag(lat_eigvals[3]), \
                np.real(lat_eigvals[4]), np.imag(lat_eigvals[4])))
    file.write(f'#A_lat_mode: \icsi = {icsi_lat}, \w_n = {wn_lat}\n')
    file.write('A_lat_w_beta = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat_w_beta[0][0], A_lat_w_beta[0][1], A_lat_w_beta[0][2], A_lat_w_beta[0][3], A_lat_w_beta[0][4],
     A_lat_w_beta[1][0], A_lat_w_beta[1][1], A_lat_w_beta[1][2], A_lat_w_beta[1][3], A_lat_w_beta[1][4],
     A_lat_w_beta[2][0], A_lat_w_beta[2][1], A_lat_w_beta[2][2], A_lat_w_beta[2][3], A_lat_w_beta[2][4],
     A_lat_w_beta[3][0], A_lat_w_beta[3][1], A_lat_w_beta[3][2], A_lat_w_beta[3][3], A_lat_w_beta[3][4],
     A_lat_w_beta[4][0], A_lat_w_beta[4][1], A_lat_w_beta[4][2], A_lat_w_beta[4][3], A_lat_w_beta[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('B_lat_w_beta = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat_w_beta[0][0], B_lat_w_beta[0][1],
     B_lat_w_beta[1][0], B_lat_w_beta[1][1],
     B_lat_w_beta[2][0], B_lat_w_beta[2][1],
     B_lat_w_beta[3][0], B_lat_w_beta[3][1],
     B_lat_w_beta[4][0], B_lat_w_beta[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    ###### TODO ######
    # define transfer function constants
    common_aero_scaling_phi = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.b
    a_phi1 = -0.5 * common_aero_scaling_phi * MAV.C_p_p * MAV.b / Va_trim
    a_phi2 = common_aero_scaling_phi * MAV.C_p_delta_a
    common_aero_scaling_theta = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.c / MAV.Jy
    a_theta1 = -0.5 * common_aero_scaling_theta * MAV.C_m_q * MAV.c / Va_trim
    a_theta2 = -common_aero_scaling_theta * MAV.C_m_alpha
    a_theta3 = common_aero_scaling_theta * MAV.C_m_delta_e

    # Compute transfer function coefficients using new propulsion model
    common_aero_scaling_theta = 0.5 * MAV.rho * Va_trim**2 * MAV.S_wing * MAV.c / MAV.Jy
    a_V1 = MAV.rho * Va_trim * MAV.S_wing / MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim + MAV.C_D_delta_e * trim_input.elevator) \
        - dT_dVa(mav, Va_trim, trim_input.throttle) / MAV.mass
    a_V2 = dT_ddelta_t(mav, Va_trim, trim_input.throttle) / MAV.mass 
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input, with_beta=False, with_alpha=False):
    x_euler = euler_state(trim_state)
    
    ##### TODO #####
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd)
    # u is the 4th state, w is the 6th state, q is the 11th state, theta is the 8th state, h is (-) the 3rd state
    A_lon = np.array([A[3, [3, 5, 10, 7, 2]],
                      A[5, [3, 5, 10, 7, 2]],
                      A[10, [3, 5, 10, 7, 2]],
                      A[7, [3, 5, 10, 7, 2]],
                      A[2, [3, 5, 10, 7, 2]]])
    B_lon = np.array([B[3, [0, 3]],
                      B[5, [0, 3]],
                      B[10, [0, 3]],
                      B[7, [0, 3]],
                      B[2, [0, 3]]])
    # change pd to h
    A_lon[-1, :-1] *= -1.
    A_lon[:, -1] *= -1.
    B_lon[-1, :] *= -1.


    # extract lateral states (v, p, r, phi, psii)
    # v is the 5th state, p is the 10th state, r is the 12th state, phi is the 7th state, psi is the 9th state
    A_lat = np.array([A[4, [4, 9, 11, 6, 8]],
                      A[9, [4, 9, 11, 6, 8]],
                      A[11, [4, 9, 11, 6, 8]],
                      A[6, [4, 9, 11, 6, 8]],
                      A[8, [4, 9, 11, 6, 8]]])
    B_lat = np.array([B[4, [1, 2]],
                      B[9, [1, 2]],
                      B[11, [1, 2]],
                      B[6, [1, 2]],
                      B[8, [1, 2]]])

    if with_beta:
        # replace v with beta in the first entry of the lateral state; see pg. 82
        mav._state = trim_state
        mav._update_velocity_data()
        Va_trim = mav._Va
        beta_trim = mav._beta
        scale = Va_trim * np.cos(beta_trim)
        A_lat[:, 0] *= scale
        A_lat[0, :] /= scale
        B_lat[0, :] /= scale

    if with_alpha:
        # replace w with alpha in the second entry of the longitudinal state; see pg. 87
        mav._state = trim_state
        mav._update_velocity_data()
        Va_trim = mav._Va
        alpha_trim = mav._beta
        scale = Va_trim * np.cos(alpha_trim)
        A_lon[:, 1] *= scale
        A_lon[1, :] /= scale
        B_lon[1, :] /= scale

    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    
    ##### TODO #####
    euler = np.array(Quaternion2Euler(x_quat[6:10])).reshape((3, 1))
    x_euler = np.zeros((12,1))
    x_euler[:6] = x_quat[:6]
    x_euler[6:9] = euler
    x_euler[9:] = x_quat[10:]
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    ##### TODO #####
    quat = Euler2Quaternion(x_euler.item(6), x_euler.item(7), x_euler.item(8))
    x_quat = np.zeros((13,1))
    x_quat[:6] = x_euler[:6]
    x_quat[6:10] = quat
    x_quat[10:] = x_euler[9:]
    return x_quat

def f_euler(mav, x_euler, delta, eps=0.001):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of Quaternion2Euler(quat) with respect to quat
    # compute partial Quaternion2Euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    ##### TODO #####
    dEuler_dquat = dxe_dxq(x_quat, eps)
    f_quat = mav._derivatives(x_quat, mav._forces_moments(delta))

    return dEuler_dquat @ f_quat

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.001  # deviation

    ##### TODO #####
    A = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            elr_left = np.copy(x_euler)
            elr_left[j, 0] -= eps
            f_left = f_euler(mav, elr_left, delta, eps=eps)
            elr_right = np.copy(x_euler)
            elr_right[j, 0] += eps
            f_right = f_euler(mav, elr_right, delta, eps=eps)
            A[i, j] = (f_right.item(i) - f_left.item(i)) / (2. * eps)
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.001  # deviation

    ##### TODO #####
    B = np.zeros((12, 4))
    delta_vec = np.array([[delta.elevator], [delta.aileron], [delta.rudder], [delta.throttle]])
    for i in range(12):
        for j in range(4):
            delta_vec_left = np.copy(delta_vec)
            delta_vec_left[j, 0] -= eps
            delta_left = MsgDelta(elevator=delta_vec_left.item(0),
                                  aileron=delta_vec_left.item(1),
                                  rudder=delta_vec_left.item(2),
                                  throttle=delta_vec_left.item(3))
            f_left = f_euler(mav, x_euler, delta_left, eps=eps)
            delta_vec_right = np.copy(delta_vec)
            delta_vec_right[j, 0] += eps
            delta_right = MsgDelta(elevator=delta_vec_right.item(0),
                                  aileron=delta_vec_right.item(1),
                                  rudder=delta_vec_right.item(2),
                                  throttle=delta_vec_right.item(3))
            f_right = f_euler(mav, x_euler, delta_right, eps=eps)
            B[i, j] = (f_right.item(i) - f_left.item(i)) / (2. * eps)
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01

    ##### TODO #####
    T_left, _ = mav._motor_thrust_torque(Va - eps, delta_t)
    T_right, _ = mav._motor_thrust_torque(Va + eps, delta_t)
    # use (second-order) central difference method
    dT_dVa = (T_right - T_left) / (2. * eps)
    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01

    ##### TODO #####
    T_left, _ = mav._motor_thrust_torque(Va, delta_t - eps)
    T_right, _ = mav._motor_thrust_torque(Va, delta_t + eps)
    # use (second-order) central difference method
    dT_ddelta_t = (T_right - T_left) / (2. * eps)
    return dT_ddelta_t

def dxe_dxq(x_quat, eps):
    # Jacobian of x_euler wrt x_quat: 12x13
    T = np.zeros((12, 13))
    T[:6, :6] = np.eye(6)
    # 4x3 matrix in the middle
    for i in range(6, 9):
        for j in range(6, 10):
            quat_left = np.copy(x_quat)
            quat_left[j, 0] -= eps
            elr_state_left = euler_state(quat_left)
            quat_right = np.copy(x_quat)
            quat_right[j, 0] += eps
            elr_state_right = euler_state(quat_right)
            T[i, j] = ( elr_state_right.item(i) - elr_state_left.item(i) ) / (2. * eps) 
    T[9:, 10:] = np.eye(3)
    return T

def convert_complex_conj_pair(lmbda, lmbda_star):
    # convert (s + \lambda)*(s + \lambda*) to s^2 + 2 * icsi * w_n * s + w_n**2
    # -> 2 * icsi * w_n = 2 * Re(\lambda)
    # -> w_n**2 = \lambda * \lambda* = |\lambda|**2
    w_n = np.real(np.sqrt(lmbda*lmbda_star))
    icsi = np.real(np.real(lmbda) / w_n)
    return icsi, w_n
