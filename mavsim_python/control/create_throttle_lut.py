import sys
sys.path.append('..')
import numpy as np
import parameters.aerosonde_parameters as MAV
from scipy.interpolate import interp1d
from scipy.optimize import minimize

THROTTLE_RES=0.02
THRUST_RES=0.2
MIN_AIRSPEED=2
MAX_AIRSPEED=45
AIRSPEED_RES=0.2

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

if __name__ == "__main__":
    throttle_vec = np.linspace(0., 1., int(1. / THROTTLE_RES) + 1)
    num_airspeed = int( (MAX_AIRSPEED - MIN_AIRSPEED) / AIRSPEED_RES ) + 1
    airspeed_vec = np.linspace(MIN_AIRSPEED, MAX_AIRSPEED, num_airspeed)

    # generate LUT that takes airspeed as input and outputs the min and max thrust at that airspeed - this will tell us when the TECS thrust command is saturating
    MIN_THRUST=1e10
    MAX_THRUST=-1e10
    for Va in airspeed_vec:
        min_thrust = calculate_motor_thrust(0., Va)
        max_thrust = calculate_motor_thrust(1., Va)
        if min_thrust < MIN_THRUST:
            MIN_THRUST = min_thrust
        if max_thrust > MAX_THRUST:
            MAX_THRUST = max_thrust

    # create a grid over the min and max thrusts - this will be used to regularize the thrust table entries
    num_thrust = int( (MAX_THRUST - MIN_THRUST) / THRUST_RES ) + 1
    thrust_vec_grid = np.linspace(MIN_THRUST, MAX_THRUST, num_thrust)
    
    # generate LUT that takes thrust and airspeed as inputs and outputs a throttle command
    with open("tecs_thrust_lut.txt", "w") as f:
        # write grid spec to first rows
        f.write(f"{MIN_THRUST}, {MAX_THRUST}, {num_thrust}\n")
        f.write(f"{MIN_AIRSPEED}, {MAX_AIRSPEED}, {num_airspeed}\n")
        i = 0
        for Va in airspeed_vec:
            thrust_vec = np.zeros_like(throttle_vec)
            for _i,th in enumerate(throttle_vec):
                thrust_vec[_i] = calculate_motor_thrust(th, Va)
            thrust_throttle = np.vstack((thrust_vec, throttle_vec))
            # order by thrust
            sorted_idx = np.argsort(thrust_throttle, axis=1)
            thrust_throttle_sorted = thrust_throttle[:, sorted_idx[0, :]]
            # create 1-d lookup and regularize
            table_1d = interp1d(thrust_throttle_sorted[0, :], thrust_throttle_sorted[1, :], kind='cubic', fill_value="extrapolate")
            j = 0
            for thrust in thrust_vec_grid:
                throttle = max(min(table_1d(thrust), 1), 0)
                f.write(f"{i}, {j}, {thrust}, {Va}, {throttle}\n")
                j += 1
            i += 1

