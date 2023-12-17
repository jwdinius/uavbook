"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Euler2Rotation
from message_types.msg_delta import MsgDelta
from models.mav_dynamics_control import MavDynamics 
import time

def compute_trim(mav: MavDynamics, Va, gamma):
    # define initial state and input

    ##### TODO #####
    # set the initial conditions of the optimization
    e0 = Euler2Quaternion(0., gamma, 0.)
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [-100],  # pd
                   [Va],  # u
                   [0.], # v
                   [0.], # w
                   [e0.item(0)],  # e0
                   [e0.item(1)],  # e1
                   [e0.item(2)],  # e2
                   [e0.item(3)],  # e3
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    mav._state = state0
    mav._update_velocity_data()
    mav._update_true_state()
    delta0 = np.array([[0],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0.5]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0).flatten()
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             },
             {'type': 'ineq',  # TODO add other actuator limits (currently only throttle is considered)
              'fun': lambda x: np.array([
                                x[16],  # delta_t >= 0
                                1 - x[16],  # delta_t <= 1
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.],
                                ])
             }
            )
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input

def trim_objective_fun(x, mav: MavDynamics, Va, gamma):
    # objective function to be minimized
    state = x[:13]
    inp = x[13:]
    ##### TODO #####
    # UPDATE mav._state so that forces and moments calculations are correct
    mav._state = state.reshape((13, 1))
    mav._update_velocity_data()
    x_star_dot = np.array([[0],  # pn_dot, don't care
                   [0],  # pe_dot, don't care
                   [Va * np.sin(gamma)],  # pd_dot
                   [0],  # u_dot
                   [0.], # v_dot
                   [0.], # w_dot
                   [0.], # e0_dot
                   [0.], # e1_dot
                   [0.], # e2_dot
                   [0.], # e3_dot
                   [0.], # p_dot
                   [0.], # q_dot
                   [0.]  # r_dot
                   ])
    # "f" in this context comes from mav._derivatives method
    delta = MsgDelta(elevator=inp[0],
                     aileron=inp[1],
                     rudder=inp[2],
                     throttle=inp[3])
    f_x_u = mav._derivatives(state, mav._forces_moments(delta))
    J = np.linalg.norm(x_star_dot[2:] - f_x_u[2:])**2
    return J
