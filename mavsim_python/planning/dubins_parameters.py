# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB
#         4/2/2020 - RWB
#         3/30/2022 - RWB

import numpy as np
import sys
sys.path.append('..')


class DubinsParameters:

    def update(self, ps, chis, pe, chie, R):
         self.p_s = ps
         self.chi_s = chis
         self.p_e = pe
         self.chi_e = chie
         self.radius = R
         self.compute_parameters()

    def compute_parameters(self):
        ps = self.p_s
        pe = self.p_e
        chis = self.chi_s
        chie = self.chi_e
        R = self.radius
        ell = np.linalg.norm(ps[0:2] - pe[0:2])

        ##### TODO #####
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than or equal to 2R.')
        else:
            # compute start and end circles
            crs = ps + R * rotz(0.5 * np.pi) @ np.array([np.cos(chis), np.sin(chis), 0]).T
            cls = ps + R * rotz(-0.5 * np.pi) @ np.array([np.cos(chis), np.sin(chis), 0]).T
            cre = pe + R * rotz(0.5 * np.pi) @ np.array([np.cos(chie), np.sin(chie), 0]).T
            cle = pe + R * rotz(-0.5 * np.pi) @ np.array([np.cos(chie), np.sin(chie), 0]).T

            north = np.array([1., 0, 0]).T
            
            # compute L1
            vartheta = compute_vartheta(cre, crs)
            l = np.linalg.norm(crs - cre)
            L1 = l + R * mod( 2 * np.pi + mod(vartheta - 0.5*np.pi) - mod(chis - 0.5*np.pi) ) + R * mod( 2 * np.pi + mod(chie - 0.5*np.pi) - mod(vartheta - 0.5*np.pi) )
            
            # compute L2
            vartheta = compute_vartheta(cle, crs)
            l = np.linalg.norm(crs - cle)
            vartheta2 = vartheta - 0.5*np.pi + np.arcsin(2*R / l) 
            L2 = np.sqrt(l**2 - 4*R**2) + R * mod( 2 * np.pi + mod(vartheta2) - mod(chis - 0.5*np.pi) ) + R * mod( 2 * np.pi + mod(vartheta2 + np.pi) - mod(chie + 0.5*np.pi) )

            # compute L3
            vartheta = compute_vartheta(cre, cls)
            l = np.linalg.norm(cre - cls)
            vartheta2 = np.arccos(2*R / l) 
            L3 = np.sqrt(l**2 - 4*R**2) + R * mod( 2 * np.pi + mod(chis + 0.5*np.pi) - mod(vartheta + vartheta2) ) + R * mod( 2 * np.pi + mod(chie - 0.5*np.pi) - mod(vartheta + vartheta2 - np.pi) )

            # compute L4
            vartheta = compute_vartheta(cle, cls)
            l = np.linalg.norm(cls - cle)
            L4 = l + R * mod( 2 * np.pi + mod(chis + 0.5*np.pi) - mod(vartheta + 0.5*np.pi) ) + R * mod( 2 * np.pi + mod(vartheta + 0.5*np.pi) - mod(chie + 0.5*np.pi) )

            # L is the minimum distance
            L = np.min([L1, L2, L3, L4])
            min_idx = np.argmin([L1, L2, L3, L4])

            if min_idx == 0:
                cs = crs
                lams = 1
                ce = cre
                lame = 1
                q1 = ce - cs
                q1 /= np.linalg.norm(q1)
                z1 = cs + R * rotz(-0.5*np.pi) @ q1
                z2 = ce + R * rotz(-0.5*np.pi) @ q1
            elif min_idx == 1:
                cs = crs
                lams = 1
                ce = cle
                lame = -1
                l = np.linalg.norm(ce - cs)
                vartheta2 = vartheta - 0.5*np.pi + np.arcsin(2 * R / l)
                q1 = rotz(vartheta2 + 0.5*np.pi) @ north
                z1 = cs + R * rotz(vartheta2) @ north
                z2 = ce + R * rotz(vartheta2 + np.pi) @ north
            elif min_idx == 2:
                cs = cls
                lams = -1
                ce = cre
                lame = 1
                vartheta2 = np.arccos(2 * R / l)
                q1 = rotz(vartheta + vartheta2 - 0.5*np.pi) @ north
                z1 = cs + R * rotz(vartheta + vartheta2) @ north
                z2 = ce + R * rotz(vartheta + vartheta2 - np.pi) @ north
            elif min_idx == 3:
                cs = cls
                lams = -1
                ce = cle
                lame = -1
                q1 = ce - cs
                q1 /= np.linalg.norm(q1)
                z1 = cs + R * rotz(0.5*np.pi) @ q1
                z2 = ce + R * rotz(0.5*np.pi) @ q1
            self.length = L
            self.center_s = cs
            self.dir_s = lams
            self.center_e = ce
            self.dir_e = lame
            self.r1 = z1
            self.n1 = q1
            self.r2 = z2
            self.r3 = pe
            self.n3 = rotz(chie) @ north

    def compute_points(self):
        ##### TODO ##### - uncomment lines and remove last line
        Del = 0.1  # distance between point

        # points along start circle
        th1 = np.arctan2(self.p_s.item(1) - self.center_s.item(1),
                         self.p_s.item(0) - self.center_s.item(0))
        th1 = mod(th1)
        th2 = np.arctan2(self.r1.item(1) - self.center_s.item(1),
                         self.r1.item(0) - self.center_s.item(0))
        th2 = mod(th2)
        th = th1
        theta_list = [th]
        if self.dir_s > 0:
            if th1 >= th2:
                while th < th2 + 2*np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2*np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)

        points = np.array([[self.center_s.item(0) + self.radius * np.cos(theta_list[0]),
                            self.center_s.item(1) + self.radius * np.sin(theta_list[0]),
                            self.center_s.item(2)]])
        for angle in theta_list:
            new_point = np.array([[self.center_s.item(0) + self.radius * np.cos(angle),
                                   self.center_s.item(1) + self.radius * np.sin(angle),
                                   self.center_s.item(2)]])
            points = np.concatenate((points, new_point), axis=0)

        # points along straight line
        sig = 0
        while sig <= 1:
            new_point = np.array([[(1 - sig) * self.r1.item(0) + sig * self.r2.item(0),
                                   (1 - sig) * self.r1.item(1) + sig * self.r2.item(1),
                                   (1 - sig) * self.r1.item(2) + sig * self.r2.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            sig += Del

        # points along end circle
        th2 = np.arctan2(self.p_e.item(1) - self.center_e.item(1),
                         self.p_e.item(0) - self.center_e.item(0))
        th2 = mod(th2)
        th1 = np.arctan2(self.r2.item(1) - self.center_e.item(1),
                         self.r2.item(0) - self.center_e.item(0))
        th1 = mod(th1)
        th = th1
        theta_list = [th]
        if self.dir_e > 0:
            if th1 >= th2:
                while th < th2 + 2 * np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2 * np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)
        for angle in theta_list:
            new_point = np.array([[self.center_e.item(0) + self.radius * np.cos(angle),
                                   self.center_e.item(1) + self.radius * np.sin(angle),
                                   self.center_e.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        return points


def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x):
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x

def compute_vartheta(pe, ps, method='v1'):
    if method == 'v1':
        return compute_vartheta_v1(pe, ps)
    else:
        return compute_vartheta_v2(pe, ps)

def compute_vartheta_v1(pe, ps):
    d = pe - ps
    return float(np.arctan2(d.item(1), d.item(0)))

def compute_vartheta_v2(pe, ps):
    d = pe - ps
    north = np.array([1., 0, 0]).T
    return float(np.arccos(north.dot(d) / np.linalg.norm(d)))

