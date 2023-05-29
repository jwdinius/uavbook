"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation


class DrawMav:
    def __init__(self, state, window):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_points()

        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
        #self.mav_body.setGLOptions('translucent')
        # ============= options include
        # opaque        Enables depth testing and disables blending
        # translucent   Enables depth testing and blending
        #               Elements must be drawn sorted back-to-front for
        #               translucency to work correctly.
        # additive      Disables depth testing, enables blending.
        #               Colors are added together, so sorting is not required.
        # ============= ======================================================
        window.addItem(self.mav_body)  # add body to plot
        # default_window_size = (500, 500)
        # window.resize(*default_window_size)


    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw MAV by resetting mesh using rotated and translated points
        self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the mav, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        ##### TODO #####
        # define MAV body parameters
        '''
        vehicle is paramaterized by:
        fuse_l1 - length from front of wing to front tip
        fuse_l2 - length from front of wing to front taper point; distance from taper point to tip is fuse_l1-fuse_l2
        fuse_l3 - length from front of wing to rear tip of vehicle
        fuse_h - height of fuselage (at its tallest point)
        fuse_w - width of fuselage (at its widest point)
        wing_l - length from back to front of wing
        wing_w - width of wing
        tailwing_l - length of tailwing
        tailwing_w - width of tailwing
        tail_h - height of the tail assembly
        
        origin of body NED is centered on the front of the wing and dropped (by fuse_h/2) into the center of the fuselage at that point
        '''
        fuse_l2 = 0.4
        fuse_l1 = 1.5 * fuse_l2
        fuse_l3 = 3 * fuse_l1
        fuse_w = fuse_l2
        fuse_h = fuse_w
        wing_l = fuse_l1
        wing_w = 4 * wing_l
        tailwing_l = 0.6 * wing_l
        tailwing_w = 2.5 * tailwing_l
        tail_h = 1.2 * fuse_h
        # exterior points (in body NED)
        # points map to index in XYZ data structure
        points = np.array([[fuse_l1, 0., 0.],
                           [fuse_l2,  0.5*fuse_w, -0.5*fuse_h],
                           [fuse_l2, -0.5*fuse_w, -0.5*fuse_h],
                           [fuse_l2, -0.5*fuse_w,  0.5*fuse_h],
                           [fuse_l2,  0.5*fuse_w,  0.5*fuse_h],
                           [-fuse_l3, 0., 0.], # fuselage
                           [0., 0.5*wing_w, 0.],
                           [-wing_l, 0.5*wing_w, 0.],
                           [-wing_l, -0.5*wing_w, 0.],
                           [0., -0.5*wing_w, 0.], # wing
                           [-fuse_l3+tailwing_l, 0.5*tailwing_w, 0.],
                           [-fuse_l3, 0.5*tailwing_w, 0.],
                           [-fuse_l3, -0.5*tailwing_w, 0.],
                           [-fuse_l3+tailwing_l, -0.5*tailwing_w, 0.], # tailwing
                           [-fuse_l3 + tailwing_l, 0., 0.],
                           [-fuse_l3, 0., -tail_h] # tail
                           ]).T

        # Define the points on the aircraft following diagram Fig 2.14
        # points are in NED coordinates
        ##### TODO #####
        #points = np.array([[0, 0, 0],  # point 1 [0]
        #                   [1, 1, 1],  # point 2 [1]
        #                   [1, 1, 0],  # point 3 [2]
        #                   ]).T

        # scale points for better rendering
        scale = 20
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)

        # Assign colors for each mesh section
        ##### TODO #####
        meshColors[:] = yellow # nose-top

        return points, meshColors

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points = points.T

        #Define each section of the mesh with 3 points
        ##### TODO #####
        #mesh = np.array([[points[0], points[1], points[2]]]) # nose-top
        mesh = np.array([
            [points[2], points[1], points[0]],
            [points[3], points[2], points[0]],
            [points[4], points[3], points[0]],
            [points[0], points[1], points[4]],
            [points[1], points[2], points[5]],
            [points[2], points[3], points[5]],
            [points[3], points[4], points[5]],
            [points[5], points[4], points[1]],  # fuselage
            [points[6], points[9], points[8]],
            [points[8], points[7], points[6]],  # wing
            [points[10], points[13], points[12]],
            [points[12], points[11], points[10]],  # tailwing
            [points[14], points[15], points[5]]   # tail
        ])
        
        return mesh
