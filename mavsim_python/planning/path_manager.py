"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - RWB
        3/30/2022 - RWB
"""

import numpy as np
import sys
sys.path.append('..')
from planning.dubins_parameters import DubinsParameters
from message_types.msg_path import MsgPath


class PathManager:
    def __init__(self):
        # message sent to path follower
        self.path = MsgPath()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()

    def update(self, waypoints, radius, state):
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
        if waypoints.type == 'straight_line':
            self.line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self.fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            self.dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        ##### TODO ######
        if self.manager_requests_waypoints is True:
            print("Don't do anything until we have waypoints")
            return

        if waypoints.flag_waypoints_changed:
            self.initialize_pointers()
            self.num_waypoints = waypoints.num_waypoints

        self.construct_line(waypoints)

        if self.inHalfSpace(mav_pos) and self.ptr_next != self.num_waypoints:
            self.increment_pointers()
            self.construct_line(waypoints)


    def fillet_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer

        ##### TODO ######
        if self.manager_requests_waypoints is True:
            print("Don't do anything until we have waypoints")
            return

        if waypoints.flag_waypoints_changed:
            self.initialize_pointers()
            self.manager_state = 1
            self.num_waypoints = waypoints.num_waypoints
        
        if self.manager_state == 1:
            self.construct_fillet_line(waypoints, radius)
            if self.inHalfSpace(mav_pos) and self.ptr_next != self.num_waypoints:
                self.manager_state = 2
        elif self.manager_state == 2:
            self.construct_fillet_circle(waypoints, radius)
            if self.inHalfSpace(mav_pos) and self.ptr_next != self.num_waypoints:
                self.increment_pointers()
                self.manager_state = 1
      

    def dubins_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer

        ##### TODO #####
        # Use functions - self.initialize_pointers(), self.dubins_path.update(),
        # self.construct_dubins_circle_start(), self.construct_dubins_line(),
        # self.inHalfSpace(), self.construct_dubins_circle_end(), self.increment_pointers(),

        # Use variables - self.num_waypoints, self.dubins_path, self.ptr_current,
        # self.ptr_previous, self.manager_state, self.manager_requests_waypoints,
        # waypoints.__, radius
        if self.manager_requests_waypoints is True:
            print("Don't do anything until we have waypoints")
            return

        if waypoints.flag_waypoints_changed:
            self.initialize_pointers()
            self.manager_state = 1
            self.num_waypoints = waypoints.num_waypoints
            ps = waypoints.ned[:, self.ptr_previous]
            chis = waypoints.course[self.ptr_previous]
            pe = waypoints.ned[:, self.ptr_current]
            chie = waypoints.course[self.ptr_current]
            self.dubins_path.update(ps, chis, pe, chie, radius)

        if self.manager_state == 1:
            self.construct_dubins_circle_start(waypoints)
            if self.inHalfSpace(mav_pos):
                self.manager_state = 2
        elif self.manager_state == 2:
            self.halfspace_n = self.dubins_path.n1.reshape((3, 1)) 
            if self.inHalfSpace(mav_pos):
                self.manager_state = 3
        elif self.manager_state == 3:
            self.construct_dubins_line(waypoints)
            if self.inHalfSpace(mav_pos):
                self.manager_state = 4
        elif self.manager_state == 4:
            self.construct_dubins_circle_end(waypoints)
            if self.inHalfSpace(mav_pos):
                self.manager_state = 5
        elif self.manager_state == 5:
            self.halfspace_n = self.dubins_path.n3.reshape((3, 1))
            if self.inHalfSpace(mav_pos) and self.ptr_next != self.num_waypoints:
                self.increment_pointers()
                self.manager_state = 1
                ps = waypoints.ned[:, self.ptr_previous]
                chis = waypoints.course[self.ptr_previous]
                pe = waypoints.ned[:, self.ptr_current]
                chie = waypoints.course[self.ptr_current]
                self.dubins_path.update(ps, chis, pe, chie, radius)
        else:
            print(f"Unknown state: {self.manager_state}")


    def initialize_pointers(self):
        if self.num_waypoints >= 3:
            ##### TODO #####
            self.ptr_previous = 0
            self.ptr_current = 1
            self.ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def increment_pointers(self):
        ##### TODO #####
        self.ptr_previous += 1
        self.ptr_current += 1
        self.ptr_next += 1

    def construct_line(self, waypoints):
        ##### TODO #####
        w_im1 = waypoints.ned[:, self.ptr_previous].reshape((3,))
        w_i = waypoints.ned[:, self.ptr_current].reshape((3,))
        if self.ptr_next != self.num_waypoints:
            w_ip1 = waypoints.ned[:, self.ptr_next].reshape((3,))
        else:
            w_ip1 = w_i

        q_im1 = (w_i - w_im1)
        q_im1 /= np.linalg.norm(q_im1)
        q_i = (w_ip1 - w_i)
        if not np.isclose( q_i.dot(q_i), 0):
            q_i /= np.linalg.norm(q_i)
        
        # update halfspace variables
        n = q_im1 + q_i
        self.halfspace_n = (n / np.linalg.norm(n)).reshape((3, 1))
        self.halfspace_r = w_i.reshape((3, 1)) 
        
        # Update path variables
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed[self.ptr_current]
        self.path.line_origin = w_im1.reshape((3, 1))
        self.path.line_direction = q_im1.reshape((3, 1))
        self.path.plot_updated = False

    def construct_fillet_line(self, waypoints, radius):
        ##### TODO #####
        w_im1 = waypoints.ned[:, self.ptr_previous].reshape((3,))
        w_i = waypoints.ned[:, self.ptr_current].reshape((3,))
        
        if self.ptr_next != self.num_waypoints:
            w_ip1 = waypoints.ned[:, self.ptr_next].reshape((3,))
        else:
            w_ip1 = w_i

        q_im1 = (w_i - w_im1)
        q_im1 /= np.linalg.norm(q_im1)
        q_i = (w_ip1 - w_i)
        if not np.isclose( q_i.dot(q_i), 0):
            q_i /= np.linalg.norm(q_i)

        varrho = np.arccos(-q_im1.dot(q_i))

        if np.isclose(varrho, 0):
            return

        # update halfspace variables
        w_i = waypoints.ned[:, self.ptr_current].reshape((3,))
        z = w_i - (radius / np.tan(0.5 * varrho) ) * q_im1 
        self.halfspace_r = z.reshape((3, 1))
        self.halfspace_n = q_im1.reshape((3, 1))

        # Update path variables
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed[self.ptr_current]
        self.path.line_origin = w_im1.reshape((3, 1))
        self.path.line_direction = q_im1.reshape((3, 1))
        self.path.plot_updated = False

    def construct_fillet_circle(self, waypoints, radius):
        ##### TODO #####
        w_im1 = waypoints.ned[:, self.ptr_previous].reshape((3,))
        w_i = waypoints.ned[:, self.ptr_current].reshape((3,))
        
        if self.ptr_next != self.num_waypoints:
            w_ip1 = waypoints.ned[:, self.ptr_next].reshape((3,))
        else:
            w_ip1 = w_i

        q_im1 = (w_i - w_im1)
        q_im1 /= np.linalg.norm(q_im1)
        q_i = (w_ip1 - w_i)
        if not np.isclose( q_i.dot(q_i), 0):
            q_i /= np.linalg.norm(q_i)

        varrho = np.arccos(-q_im1.dot(q_i))
        
        if np.isclose(varrho, 0):
            return

        qd = q_im1 - q_i
        qd /= np.linalg.norm(qd)
        c = w_i - ( radius / np.sin(0.5 * varrho) ) * qd
        lam = np.sign(q_im1[0] * q_i[1] - q_im1[1] * q_i[0])
        z = w_i + ( radius / np.tan(0.5 * varrho) ) * q_i

        self.halfspace_r = z.reshape((3, 1))
        self.halfspace_n = q_i.reshape((3, 1))
        
        # Update path variables
        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed[self.ptr_current]
        self.path.orbit_center = c.reshape((3, 1))
        self.path.orbit_radius = radius
        self.path.orbit_direction = "CCW" if lam > 0 else "CW"
        self.path.plot_updated = False

    def construct_dubins_circle_start(self, waypoints):
        ##### TODO #####
        # update halfspace variables
        self.halfspace_n = -self.dubins_path.n1.reshape((3, 1)) 
        self.halfspace_r = self.dubins_path.r1.reshape((3, 1)) 
        
        # Update path variables
        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed[self.ptr_current]
        self.path.orbit_center = self.dubins_path.center_s.reshape((3, 1))
        self.path.orbit_radius = self.dubins_path.radius
        self.path.orbit_direction = "CCW" if self.dubins_path.dir_s > 0 else "CW"
        self.path.plot_updated = False

    def construct_dubins_line(self, waypoints):
        ##### TODO #####
        # update halfspace variables
        self.halfspace_r = self.dubins_path.r2.reshape((3, 1))
        self.halfspace_n = self.dubins_path.n1.reshape((3, 1))

        # Update path variables
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed[self.ptr_current]
        self.path.line_origin = self.dubins_path.r1.reshape((3, 1))
        self.path.line_direction = self.dubins_path.n1.reshape((3, 1))
        self.path.plot_updated = False

    def construct_dubins_circle_end(self, waypoints):
        ##### TODO #####
        # update halfspace variables
        self.halfspace_n = -self.dubins_path.n3.reshape((3, 1))
        self.halfspace_r = self.dubins_path.r3.reshape((3, 1))
        
        # Update path variables
        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed[self.ptr_current]
        self.path.orbit_center = self.dubins_path.center_e.reshape((3, 1))
        self.path.orbit_radius = self.dubins_path.radius
        self.path.orbit_direction = "CCW" if self.dubins_path.dir_e > 0 else "CW"
        self.path.plot_updated = False

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

