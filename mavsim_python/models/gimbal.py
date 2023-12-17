"""
point_gimbal
    - point gimbal at target
part of mavsim
    - Beard & McLain, PUP, 2012
    - Update history:  
        3/31/2022 - RWB
"""
import numpy as np
from tools.rotations import Euler2Rotation
import parameters.camera_parameters as CAM


class Gimbal:
    def pointAtGround(self, mav):
        ###### TODO #######
        # desired inertial frame vector points down
        
        # rotate line-of-sight vector into body frame and normalize
        p_obj_i = np.array([[mav.north], [mav.east], [0]])
        return self.pointAtPosition(mav, p_obj_i)

    def pointAtPosition(self, mav, target_position):
        ###### TODO #######
        # line-of-sight vector in the inertial frame
        p_obj_i = target_position.reshape((3, 1))
        p_mav_i = np.array([[mav.north], [mav.east], [-mav.altitude]])
        l_i = p_obj_i - p_mav_i
        
        # rotate line-of-sight vector into body frame and normalize
        R_b2i = Euler2Rotation(mav.phi, mav.theta, mav.psi)
        l_r = R_b2i.T @ l_i / np.linalg.norm(l_i)
        return( self.pointAlongVector(l_r, mav.camera_az, mav.camera_el) )

    def pointAlongVector(self, ell, azimuth, elevation):
        # point gimbal so that optical axis aligns with unit vector ell
        # ell is assumed to be aligned in the body frame
        # given current azimuth and elevation angles of the gimbal

        ##### TODO #####
        # compute control inputs to align gimbal
        c_az = np.arctan2(float(ell.item(1)), float(ell.item(0)))
        c_el = np.arcsin(-float(ell.item(2)))
        # proportional control for gimbal
        u_az = CAM.k_az * (c_az - azimuth)
        u_el = CAM.k_el * (c_el - elevation)
        return( np.array([[u_az], [u_el]]) )

